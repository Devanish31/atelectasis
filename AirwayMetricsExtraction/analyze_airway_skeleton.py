import numpy as np
import nibabel as nib
import pyvista as pv
from scipy import ndimage
from skimage.morphology import skeletonize
from collections import defaultdict
from pathlib import Path


def transform_coords_for_visualization(coords, voxel_spacing, affine, skeleton_coords_reference=None):
    """
    Transform voxel coordinates to physical RAS+ space for visualization.
    
    Parameters:
    -----------
    coords : ndarray or list
        Coordinates in voxel space to transform
    voxel_spacing : tuple
        Voxel spacing (x, y, z) in mm
    affine : ndarray
        NIfTI affine matrix
    skeleton_coords_reference : ndarray, optional
        Reference skeleton coords for computing max values for flipping.
        If None, uses coords itself.
    
    Returns:
    --------
    ndarray : Coordinates in physical RAS+ space (mm)
    """
    if coords is None or len(coords) == 0:
        return coords
    
    # Handle list of tuples
    if isinstance(coords, list):
        coords = np.array(coords)
    
    # Step 1: Apply physical spacing
    coords_mm = coords.astype(np.float32) * np.array(voxel_spacing)
    
    # Step 2: Determine max values for flipping
    # Use reference skeleton if provided, otherwise use current coords
    if skeleton_coords_reference is not None:
        reference_mm = skeleton_coords_reference.astype(np.float32) * np.array(voxel_spacing)
        max_x = reference_mm[:, 0].max()
        max_y = reference_mm[:, 1].max()
        max_z = reference_mm[:, 2].max()
    else:
        max_x = coords_mm[:, 0].max()
        max_y = coords_mm[:, 1].max()
        max_z = coords_mm[:, 2].max()
    
    # Step 3: Apply axis flips based on affine matrix
    x_flip = affine[0, 0] < 0
    y_flip = affine[1, 1] < 0
    z_flip = affine[2, 2] < 0
    
    if x_flip:
        coords_mm[:, 0] = max_x - coords_mm[:, 0]
    
    if y_flip:
        coords_mm[:, 1] = max_y - coords_mm[:, 1]
    
    if z_flip:
        coords_mm[:, 2] = max_z - coords_mm[:, 2]
    
    return coords_mm


def calculate_perpendicular_cross_section(point, direction_vector, airway_mask, voxel_spacing, max_radius_mm=10):
    """
    Calculate cross-sectional area perpendicular to the bronchus axis at a given point.
    
    Parameters:
    - point: (z, y, x) coordinates of the centerline point
    - direction_vector: normalized direction vector of the bronchus at this point
    - airway_mask: 3D binary mask of the airway segmentation
    - voxel_spacing: (z_spacing, y_spacing, x_spacing) in mm
    - max_radius_mm: maximum radius to search for airway boundary
    
    Returns:
    - cross_section_area_mm2: cross-sectional area in mm²
    """
    point = np.array(point)
    direction_vector = np.array(direction_vector)
    voxel_spacing = np.array(voxel_spacing)
    
    # Normalize direction vector
    direction_norm = direction_vector / np.linalg.norm(direction_vector)
    
    # Create two orthogonal vectors perpendicular to the direction
    if abs(direction_norm[2]) < 0.9:
        arbitrary = np.array([0, 0, 1])
    else:
        arbitrary = np.array([0, 1, 0])
    
    # First perpendicular vector (in voxel space)
    perp1_voxel = np.cross(direction_norm, arbitrary)
    perp1_voxel = perp1_voxel / np.linalg.norm(perp1_voxel)
    
    # Second perpendicular vector (in voxel space)
    perp2_voxel = np.cross(direction_norm, perp1_voxel)
    perp2_voxel = perp2_voxel / np.linalg.norm(perp2_voxel)
    
    # Convert perpendicular vectors to physical space
    perp1_physical = perp1_voxel * voxel_spacing
    perp2_physical = perp2_voxel * voxel_spacing
    
    # Physical lengths of perpendicular vectors (mm)
    perp1_length = np.linalg.norm(perp1_physical)
    perp2_length = np.linalg.norm(perp2_physical)
    
    # Normalize physical perpendicular vectors
    perp1_physical_norm = perp1_physical / perp1_length
    perp2_physical_norm = perp2_physical / perp2_length
    
    # Average physical spacing in the perpendicular plane
    avg_perp_spacing = (perp1_length + perp2_length) / 2
    
    # Sample points in the perpendicular plane
    max_radius_voxels = max_radius_mm / avg_perp_spacing
    
    # Count airway voxels in the perpendicular plane
    airway_voxel_count = 0
    
    # Sample in a grid pattern in the perpendicular plane
    sample_step = 0.5  # Sample every 0.5 voxels in the plane
    for r in np.arange(0, max_radius_voxels, sample_step):
        for theta in np.linspace(0, 2*np.pi, int(2*np.pi*r/sample_step) + 1):
            # Point in perpendicular plane (voxel space)
            sample_point = point + r * (np.cos(theta) * perp1_voxel + np.sin(theta) * perp2_voxel)
            
            # Round to nearest voxel
            sample_coords = tuple(np.round(sample_point).astype(int))
            
            # Check if within bounds
            if (0 <= sample_coords[0] < airway_mask.shape[0] and
                0 <= sample_coords[1] < airway_mask.shape[1] and
                0 <= sample_coords[2] < airway_mask.shape[2]):
                
                if airway_mask[sample_coords] > 0:
                    airway_voxel_count += 1
    
    # Calculate physical area per sample point
    # The area is based on the physical spacing in the perpendicular plane
    voxel_area_mm2 = (sample_step * avg_perp_spacing)**2
    cross_section_area_mm2 = airway_voxel_count * voxel_area_mm2
    
    return cross_section_area_mm2

def analyze_right_lung_airways(nii_file_path, carina_zone_radius_cm=1, lookahead_distance=10, visualize=True, screenshot_path=None, title=None):
    """
    Analyze right lung airways and identify anatomical structures using bifurcation zone approach.
    
    Parameters:
    -----------
    nii_file_path : str
        Path to the NIfTI airway segmentation file
    carina_zone_radius_cm : float
        Radius of bifurcation zone for carina (in cm, default: 1)
    lookahead_distance : int
        Number of points to look ahead for branch direction (default: 10)
    visualize : bool
        Whether to create interactive 3D visualization (default: True)
    
    Returns:
    --------
    dict containing anatomical structures and analysis results
    """
    
    print("="*60)
    print(f"ANALYZING AIRWAYS: {nii_file_path}")
    print("="*60)

    # Load and skeletonize
    nii_img = nib.load(nii_file_path)
    airway_mask = nii_img.get_fdata().astype(bool)
    voxel_spacing = nii_img.header.get_zooms()
    affine = nii_img.affine

    print(f"Shape: {airway_mask.shape}")
    print(f"Voxel spacing (mm): {voxel_spacing}")

    # Skeletonize
    print("\nSkeletonizing...")
    skeleton = skeletonize(airway_mask)
    skeleton_coords_original = np.array(np.where(skeleton)).T

    # ============================================
    # EXCLUDE PERIPHERAL 6% OF SKELETON EXTENT
    # ============================================
    print("\nExcluding peripheral skeleton points...")

    # Get skeleton bounds FROM ORIGINAL
    x_min_skel = skeleton_coords_original[:, 0].min()
    x_max_skel = skeleton_coords_original[:, 0].max()
    y_min_skel = skeleton_coords_original[:, 1].min()
    y_max_skel = skeleton_coords_original[:, 1].max()

    x_extent = x_max_skel - x_min_skel
    y_extent = y_max_skel - y_min_skel

    # Calculate 6% exclusion boundaries
    x_excl_left = x_min_skel + (x_extent * 0.06)
    x_excl_right = x_max_skel - (x_extent * 0.06)
    y_excl_anterior = y_min_skel + (y_extent * 0.06)
    y_excl_posterior = y_max_skel - (y_extent * 0.06)

    print(f"  Original skeleton points: {len(skeleton_coords_original)}")
    print(f"  X bounds: {x_min_skel:.1f} to {x_max_skel:.1f} (extent: {x_extent:.1f})")
    print(f"  Y bounds: {y_min_skel:.1f} to {y_max_skel:.1f} (extent: {y_extent:.1f})")
    print(f"  Keeping X: {x_excl_left:.1f} to {x_excl_right:.1f}")
    print(f"  Keeping Y: {y_excl_anterior:.1f} to {y_excl_posterior:.1f}")

    # Filter to keep only central 88%
    central_mask = (
        (skeleton_coords_original[:, 0] >= x_excl_left) &
        (skeleton_coords_original[:, 0] <= x_excl_right) &
        (skeleton_coords_original[:, 1] >= y_excl_anterior) &
        (skeleton_coords_original[:, 1] <= y_excl_posterior)
    )

    # NOW use the filtered version for everything downstream
    skeleton_coords = skeleton_coords_original[central_mask]

    print(f"  Filtered skeleton points: {len(skeleton_coords)}")
    print(f"  Excluded {len(skeleton_coords_original) - len(skeleton_coords)} peripheral points")
    # ============================================

    # Build skeleton graph
    skeleton_dict = defaultdict(list)
    skeleton_set = set(map(tuple, skeleton_coords))
    
    for coord in skeleton_coords:
        x, y, z = coord
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (x+dx, y+dy, z+dz)
                    if neighbor in skeleton_set:
                        skeleton_dict[tuple(coord)].append(neighbor)

    def count_connected_component_size(start_point, skeleton_dict, max_search=1000):
        """
        Count the size of the connected component containing start_point.
        Uses BFS to find all connected skeleton points.
        """
        visited = set()
        queue = [start_point]
        visited.add(start_point)
        
        while queue and len(visited) < max_search:
            current = queue.pop(0)
            
            # Add all unvisited neighbors
            for neighbor in skeleton_dict.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited)
    # ============================================
    
    # Find bifurcations (filter by exclusion zone)
    bifurcations = [coord for coord, neighbors in skeleton_dict.items() if len(neighbors) >= 3]
    bifurcations_set = set(bifurcations)

    print(f"Total bifurcations: {len(bifurcations)}")
    
    # ============================================
    # NEW: Helper function to find nearest skeleton point (for gap jumping)
    # ============================================
    def find_nearest_unvisited_skeleton_point(current_point, visited_set, skeleton_coords, max_distance=3, prefer_inferior=False, inferior_bias=2.0):
        """
        Find nearest unvisited skeleton point within max_distance voxels.
        Returns None if no point found.
        
        Parameters:
        -----------
        current_point : tuple
            Current point coordinates
        visited_set : set
            Set of visited points
        skeleton_coords : ndarray
            Array of all skeleton coordinates
        max_distance : int or float
            Maximum distance in voxels
        prefer_inferior : bool
            If True, prefer points with lower Z values (inferior direction)
        inferior_bias : float
            Multiplier for inferior preference (higher = stronger preference)
        """
        current_arr = np.array(current_point)
        
        # Get all unvisited skeleton points
        unvisited_coords = [coord for coord in skeleton_coords if tuple(coord) not in visited_set]
        
        if len(unvisited_coords) == 0:
            return None
        
        unvisited_arr = np.array(unvisited_coords)
        
        # Calculate distances
        distances = np.sqrt(np.sum((unvisited_arr - current_arr)**2, axis=1))
        
        # Filter by max_distance
        within_range_mask = distances <= max_distance
        
        if not np.any(within_range_mask):
            return None
        
        # Apply inferior preference if requested
        if prefer_inferior:
            # Calculate Z difference (negative = inferior/downward)
            z_diff = unvisited_arr[:, 2] - current_arr[2]
            
            # Create penalty for superior points, bonus for inferior points
            # Positive z_diff (going up) gets penalized, negative (going down) gets bonus
            z_penalty = np.where(z_diff > 0, z_diff * inferior_bias, -z_diff / inferior_bias)
            
            # Adjust distances with Z penalty
            adjusted_distances = distances + z_penalty
            
            # Find closest within max_distance using adjusted distances
            adjusted_distances[~within_range_mask] = np.inf
            min_dist_idx = np.argmin(adjusted_distances)
        else:
            # Original behavior - just find closest
            distances[~within_range_mask] = np.inf
            min_dist_idx = np.argmin(distances)
        
        # Return the point if it's within range
        if within_range_mask[min_dist_idx]:
            return tuple(unvisited_coords[min_dist_idx])
        
        return None
    # ============================================

    # ============================================
    # NEW: Find center-most superior point (within central 15% diameter)
    # ============================================
    print("\nFinding trachea starting point...")

    # Calculate center based on ACTUAL skeleton coordinates (not array shape)
    image_center_x = np.mean(skeleton_coords[:, 0])
    image_center_y = np.mean(skeleton_coords[:, 1])

    # Calculate radius as percentage of skeleton extent (not array shape)
    x_extent = skeleton_coords[:, 0].max() - skeleton_coords[:, 0].min()
    y_extent = skeleton_coords[:, 1].max() - skeleton_coords[:, 1].min()

    max_radius_x = x_extent * 0.16
    max_radius_y = y_extent * 0.40 

    print(f"  Skeleton center: ({image_center_x:.1f}, {image_center_y:.1f})")
    print(f"  Skeleton extent: X={x_extent:.1f}, Y={y_extent:.1f}")
    print(f"  Search radius: ({max_radius_x:.1f}, {max_radius_y:.1f}) voxels")

    # Calculate Z-range exclusion (skip top 14%)
    z_coords = skeleton_coords[:, 2]
    z_min = z_coords.min()
    z_max = z_coords.max()
    z_range = z_max - z_min
    z_threshold = z_max - (0.14 * z_range)

    print(f"  Z-range: {z_min:.1f} to {z_max:.1f} (range: {z_range:.1f})")
    print(f"  Starting search from Z={z_threshold:.1f} (excluding top 14%)")

    # Filter skeleton points within central region AND below exclusion threshold
    central_mask = (
        (np.abs(skeleton_coords[:, 0] - image_center_x) <= max_radius_x) &
        (np.abs(skeleton_coords[:, 1] - image_center_y) <= max_radius_y) &
        (skeleton_coords[:, 2] <= z_threshold)
    )
    central_skeleton_coords = skeleton_coords[central_mask]

    MIN_COMPONENT_SIZE = 55

    if len(central_skeleton_coords) == 0:
        print("  ⚠ WARNING: No skeleton points in central region below exclusion zone")
        below_threshold = skeleton_coords[skeleton_coords[:, 2] <= z_threshold]
        if len(below_threshold) > 0:
            most_superior = max(below_threshold, key=lambda p: p[2])
            print(f"  Using global search below threshold")
        else:
            print("  ⚠ WARNING: No points below threshold, using absolute max")
            most_superior = max(skeleton_coords, key=lambda p: p[2])
    else:
        # Find most superior point within central region, filtering out small islands
        max_z = central_skeleton_coords[:, 2].max()
        z_tolerance = 2
        
        # Get all points near the maximum Z
        near_max_z = central_skeleton_coords[central_skeleton_coords[:, 2] >= (max_z - z_tolerance)]
        
        # Calculate X-distance to center only (ignore Y)
        x_distances_to_center = np.abs(near_max_z[:, 0] - image_center_x)
        
        # Sort by distance to center (closest first)
        sorted_indices = np.argsort(x_distances_to_center)
        
        # Try candidates in order of proximity to center
        valid_starting_point_found = False
        for idx in sorted_indices:
            candidate_point = tuple(near_max_z[idx])
            
            # Count connected component size for this candidate
            component_size = count_connected_component_size(candidate_point, skeleton_dict, max_search=1000)
            
            print(f"  Testing candidate {candidate_point}: component size = {component_size} voxels")
            
            if component_size >= MIN_COMPONENT_SIZE:
                most_superior = candidate_point
                valid_starting_point_found = True
                print(f"  ✓ Valid starting point found!")
                print(f"  ✓ Component size: {component_size} voxels (≥ {MIN_COMPONENT_SIZE})")
                print(f"  ✓ X-distance to cylinder center: {x_distances_to_center[idx]:.1f} voxels")
                break
            else:
                print(f"  ✗ Component too small ({component_size} < {MIN_COMPONENT_SIZE}), trying next candidate...")
        
        if not valid_starting_point_found:
            print(f"  ⚠ WARNING: No candidates with component size ≥ {MIN_COMPONENT_SIZE}")
            print(f"  Falling back to most superior point regardless of component size")
            most_central_idx = sorted_indices[0]
            most_superior = near_max_z[most_central_idx]
        
        print(f"  ✓ Found {len(central_skeleton_coords)} skeleton points in search region")
        print(f"  ✓ Tested {len(near_max_z)} candidates within {z_tolerance} voxels of max Z")

    print(f"  Starting point: {most_superior}")

    # ============================================
    # NEW: Mark exclusion zone as visited to prevent lookaheads from going there
    # ============================================
    exclusion_zone_visited = set()
    for coord in skeleton_coords:
        if coord[2] > z_threshold:  # Above the threshold = in exclusion zone
            exclusion_zone_visited.add(tuple(coord))

    print(f"  Marked {len(exclusion_zone_visited)} skeleton points in exclusion zone as pre-visited")
    # ============================================
    
    # Trace function
    def trace_from_point(start_point, visited_set, skeleton_dict, skeleton_coords, max_length=100, allow_gap_jump=True, max_gap_distance=3, max_gap_jumps=3, prefer_inferior_jumps=False):
        """
        Trace from a point with optional gap-jumping capability.
        
        Parameters:
        -----------
        start_point : tuple
            Starting point coordinates
        visited_set : set
            Set of already visited points
        skeleton_dict : dict
            Dictionary of skeleton connectivity
        skeleton_coords : ndarray
            Array of all skeleton coordinates (needed for gap jumping)
        max_length : int
            Maximum path length
        allow_gap_jump : bool
            Whether to allow jumping small gaps (default: True)
        max_gap_distance : int
            Maximum distance in voxels to jump (default: 3)
        max_gap_jumps : int
            Maximum number of gap jumps allowed (default: 3)
        prefer_inferior_jumps : bool
            Whether to prefer inferior (downward) jumps when gap jumping
        """
        path = [start_point]
        visited_local = visited_set | {start_point}
        current = start_point
        gap_jumps_used = 0
        
        for step in range(max_length):
            next_neighbors = [n for n in skeleton_dict[current] if n not in visited_local]
            
            # If no neighbors, try gap jumping if allowed
            if len(next_neighbors) == 0:
                if allow_gap_jump and gap_jumps_used < max_gap_jumps:
                    nearest_point = find_nearest_unvisited_skeleton_point(
                        current, visited_local, skeleton_coords, 
                        max_distance=max_gap_distance,
                        prefer_inferior=prefer_inferior_jumps,
                        inferior_bias=2.0
                    )
                    
                    if nearest_point is not None:
                        path.append(nearest_point)
                        visited_local.add(nearest_point)
                        current = nearest_point
                        gap_jumps_used += 1
                        continue
                    else:
                        # No nearby unvisited point found
                        break
                else:
                    # No gap jumping allowed or limit reached
                    break
            
            # Select next point based on direction
            if len(path) > 1:
                prev_point = np.array(path[-2])
                curr_point = np.array(current)
                direction = curr_point - prev_point
                
                best_neighbor = None
                best_score = -np.inf
                
                for neighbor in next_neighbors:
                    neighbor_direction = np.array(neighbor) - curr_point
                    score = np.dot(direction, neighbor_direction)
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
                
                next_point = best_neighbor
            else:
                next_point = next_neighbors[0]
            
            path.append(next_point)
            visited_local.add(next_point)
            current = next_point
        
        return path
    
    # Find trachea and identify carina iteratively
    trachea_skeleton = []
    current = tuple(most_superior)
    visited_trachea = {current} | exclusion_zone_visited  # Include exclusion zone!
    trachea_skeleton.append(current)

    print("\nTracing trachea and searching for carina...")

    carina_found = False
    attempt = 0
    gap_jumps = 0
    max_gap_jumps = 10  # Allow up to 10 gap jumps total

    while not carina_found and attempt < 10:
        # ============================================
        # Trace trachea with gap jumping capability
        # ============================================
        for step in range(300):
            neighbors = [n for n in skeleton_dict[current] if n not in visited_trachea]
            
            # If no neighbors, try to jump gap
            if len(neighbors) == 0:
                if gap_jumps < max_gap_jumps:

                    nearest_point = find_nearest_unvisited_skeleton_point(current, visited_trachea, skeleton_coords, max_distance=20)
                    if nearest_point is not None:
                        gap_distance = np.sqrt(np.sum((np.array(nearest_point) - np.array(current))**2))
                        print(f"  ↷ Gap jump #{gap_jumps + 1}: {gap_distance:.1f} voxels from {current} to {nearest_point}")
                        
                        # Jump to the nearest point
                        trachea_skeleton.append(nearest_point)
                        visited_trachea.add(nearest_point)
                        current = nearest_point
                        gap_jumps += 1
                        continue
                    else:
                        print(f"  ✗ No skeleton points within 20 voxels, stopping (after {gap_jumps} gap jumps)")
                        break
                else:
                    print(f"  ✗ Max gap jumps ({max_gap_jumps}) reached, stopping")
                    break
            
            if current in bifurcations_set and len(neighbors) >= 2:
                print(f"  >> Found bifurcation at {current} with {len(neighbors)} neighbors (step {step})")
                break
            
            # Select next point based on direction
            if len(trachea_skeleton) > 1:
                prev_point = np.array(trachea_skeleton[-2])
                curr_point = np.array(current)
                direction = curr_point - prev_point
                
                best_neighbor = None
                best_score = -np.inf
                for n in neighbors:
                    n_dir = np.array(n) - curr_point
                    score = np.dot(direction, n_dir)
                    if score > best_score:
                        best_score = score
                        best_neighbor = n
                next_point = best_neighbor
            else:
                next_point = min(neighbors, key=lambda p: (p[2], p[1]))   ## Can remove p[1]
                print(f"  Step 0: Selected initial direction - from {current} to {next_point}")
                print(f"         Z change: {next_point[2] - current[2]}")
            
            trachea_skeleton.append(next_point)
            visited_trachea.add(next_point)
            current = next_point
        # ============================================
        
        # If we didn't find a bifurcation, check if path is too short (isolated noise)                  ## CAN REMOVE THIS??
        if current not in bifurcations_set:
            total_path_length = len(visited_trachea) - len(exclusion_zone_visited)
            
            if total_path_length <= 5:
                print(f"  ✗ Dead end with only {total_path_length} traversed points - likely isolated noise")
                print(f"  Trying next highest point in central region...")
                
                used_points = visited_trachea - exclusion_zone_visited
                remaining_central = [p for p in central_skeleton_coords if tuple(p) not in used_points]
                
                if len(remaining_central) > 0:
                    most_superior = max(remaining_central, key=lambda p: p[2])
                    print(f"  New starting point: {most_superior}")
                    
                    trachea_skeleton = []
                    current = tuple(most_superior)
                    visited_trachea = {current} | exclusion_zone_visited
                    trachea_skeleton.append(current)
                    gap_jumps = 0
                    attempt = 0
                    continue
                else:
                    print(f"  ✗ No more starting points available")
                    break
            else:
                print(f"  ✗ Dead end after {total_path_length} traversed points")
                break
        
        first_trachea_bifurcation = current
        attempt += 1
        print(f"\nAttempt {attempt}: Testing bifurcation at {first_trachea_bifurcation}")
        
        # Create bifurcation zone around this bifurcation
        zone_radius_mm = carina_zone_radius_cm * 10
        carinal_zone_center = np.array(first_trachea_bifurcation)
        
        print(f"  CARINAL ZONE:")
        
        # Find all bifurcations within the zone
        carinal_zone_bifurcations = []
        for bif in bifurcations:
            bif_arr = np.array(bif)
            dist_mm = np.sqrt(np.sum(((bif_arr - carinal_zone_center) * voxel_spacing)**2))
            if dist_mm <= zone_radius_mm:
                carinal_zone_bifurcations.append(bif)
        
        print(f"  Bifurcations in zone: {len(carinal_zone_bifurcations)}")
        
        # Find all skeleton points in the zone
        carinal_zone_points = []
        for coord in skeleton_coords:
            coord_arr = np.array(coord)
            dist_mm = np.sqrt(np.sum(((coord_arr - carinal_zone_center) * voxel_spacing)**2))
            if dist_mm <= zone_radius_mm:
                carinal_zone_points.append(tuple(coord))
        
        carinal_zone_points_set = set(carinal_zone_points)
        print(f"  Total skeleton points in zone: {len(carinal_zone_points)}")
        
        # Find branches emerging from the zone
        carinal_zone_exit_points = []
        for zone_point in carinal_zone_points:
            neighbors = skeleton_dict[zone_point]
            for neighbor in neighbors:
                if neighbor not in carinal_zone_points_set and neighbor not in visited_trachea:
                    carinal_zone_exit_points.append({
                        'zone_point': zone_point,
                        'exit_point': neighbor
                    })
        
        print(f"  Exit points from zone: {len(carinal_zone_exit_points)}")

        # For each exit point, trace a lookahead path
        visited_carinal_zone_and_trachea = visited_trachea | carinal_zone_points_set

        carinal_branch_candidates = []
        for exit_info in carinal_zone_exit_points:

            CARINA_LOOKAHEAD_DISTANCE = 10

            lookahead_path = trace_from_point(
                exit_info['exit_point'], 
                visited_carinal_zone_and_trachea, 
                skeleton_dict,
                skeleton_coords,
                max_length=CARINA_LOOKAHEAD_DISTANCE + 5
            )
            
            if len(lookahead_path) >= CARINA_LOOKAHEAD_DISTANCE:
                distal_point = lookahead_path[CARINA_LOOKAHEAD_DISTANCE - 1]
                mean_x = np.mean([p[0] for p in lookahead_path[:CARINA_LOOKAHEAD_DISTANCE]])
                
                print(f"      ✓ ACCEPTED as valid branch")
                
                carinal_branch_candidates.append({
                    'zone_point': exit_info['zone_point'],
                    'exit_point': exit_info['exit_point'],
                    'lookahead_path': lookahead_path[:CARINA_LOOKAHEAD_DISTANCE],
                    'distal_point': distal_point,
                    'mean_x': mean_x
                })
            else:
                print(f"      ✗ REJECTED - too short")
        
        print(f"  Branch candidates with valid lookahead: {len(carinal_branch_candidates)}")
        
        # Check if we have at least 2 candidates
        if len(carinal_branch_candidates) >= 2:
            print(f"  ✓ Found valid carina with {len(carinal_branch_candidates)} branches!")
            carina_found = True
        else:
            print(f"  ✗ Insufficient branches, continuing to next bifurcation...")
            
            # Mark all zone points as part of trachea                              
            print(f"  Marking {len(carinal_zone_points)} zone points as trachea")
            for zone_point in carinal_zone_points:
                if zone_point not in visited_trachea:
                    trachea_skeleton.append(zone_point)
                    visited_trachea.add(zone_point)
            
            # If there's a single branch candidate, follow it as trachea continuation
            if len(carinal_branch_candidates) == 1:
                exit_point = carinal_branch_candidates[0]['exit_point']
                print(f"  Following single branch from zone exit: {exit_point}")
                
                # Trace from exit point, but STOP at the next bifurcation
                continuation_path = []
                trace_current = exit_point
                trace_visited = visited_trachea | carinal_zone_points_set
                
                for trace_step in range(100):
                    continuation_path.append(trace_current)
                    trace_visited.add(trace_current)
                    
                    # Check if we hit a bifurcation - STOP HERE
                    if trace_current in bifurcations_set:
                        print(f"  Hit next bifurcation at {trace_current} after {len(continuation_path)} points")
                        break
                    
                    # Get next neighbors
                    next_neighbors = [n for n in skeleton_dict[trace_current] if n not in trace_visited]
                    if len(next_neighbors) == 0:
                        break
                    
                    # Select next point based on direction
                    if len(continuation_path) > 1:
                        prev = np.array(continuation_path[-2])
                        curr = np.array(trace_current)
                        direction = curr - prev
                        
                        best_neighbor = None
                        best_score = -np.inf
                        for n in next_neighbors:
                            n_dir = np.array(n) - curr
                            score = np.dot(direction, n_dir)
                            if score > best_score:
                                best_score = score
                                best_neighbor = n
                        trace_current = best_neighbor
                    else:
                        trace_current = min(next_neighbors, key=lambda p: (p[2], p[1]))
                
                # Add continuation to trachea
                for point in continuation_path:
                    if point not in visited_trachea:
                        trachea_skeleton.append(point)
                        visited_trachea.add(point)
                
                # Set current to last point (should be a bifurcation)
                if len(continuation_path) > 0:
                    current = continuation_path[-1]
                    print(f"  Continued {len(continuation_path)} points to {current}")
                else:
                    break
                

    # Initialize variables that might not be set
    right_bifurcations = []
    rul_zone_center = None
    int_zone_center = None
    rml_found = False

    if not carina_found:
        print("\n⚠ WARNING: Could not find valid carina with 2 branches!")
        right_main_start = None
        left_main_start = None
    else:
        print(f"\n✓ CARINA IDENTIFIED at: {first_trachea_bifurcation}")
        
        # Classify the branches
        if len(carinal_branch_candidates) >= 2:
            carinal_branch_candidates.sort(key=lambda x: x['mean_x'])
            
            right_branch_info = carinal_branch_candidates[0]
            left_branch_info = carinal_branch_candidates[-1]
            
            right_main_start = right_branch_info['zone_point']
            left_main_start = left_branch_info['zone_point']
            
            print(f"  Right main starting from zone point: {right_main_start}")
            print(f"    Mean X of lookahead: {right_branch_info['mean_x']:.1f}")
            print(f"  Left main starting from zone point: {left_main_start}")
            print(f"    Mean X of lookahead: {left_branch_info['mean_x']:.1f}")
    
    # Trace right main bronchus from the zone point
    right_main_skeleton = None
    rul_skeleton = None
    intermedius_skeleton = None
    rml_skeleton = None
    rll_skeleton = None
    
    if right_main_start:
        print(f"\nTracing right main bronchus from {right_main_start}...")
        right_main_skeleton = []
        current = right_main_start
        visited_right = visited_trachea | carinal_zone_points_set | {current}
        right_main_skeleton.append(current)
        right_bifurcations = []

        MIN_RIGHT_MAIN_LENGTH = 10  # voxels
        
        for step in range(100):
            neighbors = [n for n in skeleton_dict[current] if n not in visited_right]

            if len(neighbors) == 0:
                if step < 20:
                    print(f"    >> No neighbors available - STOPPING")
                break
            
            # Check if current point is a bifurcation
            if current in bifurcations_set:
                # Calculate distance from right main start
                distance_from_start = np.linalg.norm(
                    np.array(current) - np.array(right_main_start)
                )
                
                if distance_from_start < MIN_RIGHT_MAIN_LENGTH:
                    print(f"    >> Found bifurcation at {current} but too close to carina ({distance_from_start:.1f} voxels)")
                    print(f"    >> Continuing through it...")
                else:
                    # Far enough from carina - this is a valid RUL bifurcation
                    right_bifurcations.append(current)
                    print(f"    >> HIT VALID BIFURCATION - STOPPING at {current} (distance: {distance_from_start:.1f} voxels)")
                    break
            
            # Continue tracing
            if len(right_main_skeleton) > 1:
                prev = np.array(right_main_skeleton[-2])
                curr = np.array(current)
                direction = curr - prev
                
                if step < 20:
                    print(f"    Using direction from {tuple(prev.astype(int))} to {current}")
                    print(f"    Direction vector: {direction}")
                
                best_neighbor = None
                best_score = -np.inf
                neighbor_scores = []
                
                for n in neighbors:
                    n_dir = np.array(n) - curr
                    score = np.dot(direction, n_dir)
                    neighbor_scores.append((n, score))
                    if score > best_score:
                        best_score = score
                        best_neighbor = n
                
                next_point = best_neighbor
                if step < 20:
                    print(f"    >> SELECTED: {next_point} (score: {best_score:.2f})")
            else:
                next_point = neighbors[0]
                if step < 20:
                    print(f"    >> SELECTED (first step): {next_point}")
            
            right_main_skeleton.append(next_point)
            visited_right.add(next_point)
            current = next_point
        
        # Continue with RUL/Intermedius analysis as before
        if len(right_bifurcations) > 0:

            rul_found = False
            
            for bif_idx, test_right_bif in enumerate(right_bifurcations):
                print(f"\nTesting RUL bifurcation {bif_idx+1}/{len(right_bifurcations)} at: {test_right_bif}")
                
                
                # Create 0.5cm bifurcation zone around this right bifurcation
                rul_zone_radius_cm = 0.5
                rul_zone_radius_mm = rul_zone_radius_cm * 10
                rul_zone_center = np.array(test_right_bif)
                
                # Find all skeleton points in RUL bifurcation zone
                rul_zone_points = []
                for coord in skeleton_coords:
                    coord_arr = np.array(coord)
                    dist_mm = np.sqrt(np.sum(((coord_arr - rul_zone_center) * voxel_spacing)**2))
                    if dist_mm <= rul_zone_radius_mm:
                        rul_zone_points.append(tuple(coord))
                
                rul_zone_points_set = set(rul_zone_points)
                print(f"  RUL ZONE: {len(rul_zone_points)} skeleton points")
                
                # Find exit points from RUL zone
                visited_right = visited_trachea | carinal_zone_points_set | set(right_main_skeleton)
                rul_exit_points = []
                
                for zone_point in rul_zone_points:
                    neighbors = skeleton_dict[zone_point]
                    for neighbor in neighbors:
                        if neighbor not in rul_zone_points_set and neighbor not in visited_right:
                            rul_exit_points.append({
                                'zone_point': zone_point,
                                'exit_point': neighbor
                            })
                
                print(f"  Exit points: {len(rul_exit_points)}")
                
                # Trace lookaheads from exit points
                visited_up_to_bif = visited_right | rul_zone_points_set
                rul_branch_candidates = []
                
                # ==== ADD DETAILED DEBUG FOR LOOKAHEAD TRACING ====
                print(f"\n  LOOKAHEAD TRACING (need ≥5 points for valid candidate):")
                for i, exit_info in enumerate(rul_exit_points):
                    print(f"    Exit {i+1} from {exit_info['zone_point']} → {exit_info['exit_point']}:")
                    
                    lookahead_path = trace_from_point(
                        exit_info['exit_point'],
                        visited_up_to_bif,
                        skeleton_dict,
                        skeleton_coords,
                        max_length=10 + 5
                    )
                    
                    print(f"      Lookahead path length: {len(lookahead_path)}")

                    # Trace the FULL branch to see total available length
                    full_branch = trace_from_point(
                        exit_info['exit_point'],
                        visited_up_to_bif,
                        skeleton_dict,
                        skeleton_coords,
                        max_length=200  # Trace as far as possible to terminal end
                    )
                    print(f"      Full branch length: {len(full_branch)} voxels")

                    if len(lookahead_path) >= 5:
                        lookahead_point = lookahead_path[4]
                        direction = np.array(lookahead_point) - rul_zone_center
                        score = direction[2] + direction[1]
                        
                        print(f"      ✓ VALID - Length: {len(lookahead_path)}")
                        print(f"        5th point: {lookahead_point}")
                        print(f"        Direction from zone center: {direction}")
                        print(f"        Score (Z+Y): {score:.1f}")
                        
                        # Show first few points of path
                        print(f"        First 5 path points: {lookahead_path[:5]}")
                        
                        rul_branch_candidates.append({
                            'zone_point': exit_info['zone_point'],
                            'exit_point': exit_info['exit_point'],
                            'lookahead_path': lookahead_path[:10],
                            'score': score
                        })
                    else:
                        print(f"      ✗ INVALID - Only {len(lookahead_path)} points (need ≥5)")
                        print(f"        Path: {lookahead_path}")
                        
                # ==================================================
                
                print(f"\n  Branch candidates with valid lookahead: {len(rul_branch_candidates)}")
                
                # Check if we have at least 2 candidates
                if len(rul_branch_candidates) >= 2:
                    print(f"  ✓ Found valid RUL bifurcation with {len(rul_branch_candidates)} branches!")
                    rul_found = True
                    break
                else:
                    print(f"  ✗ Insufficient branches, continuing to next bifurcation...")
                    
                    # Mark all zone points as part of right main bronchus                                 
                    print(f"  Marking {len(rul_zone_points)} zone points as right main")
                    for zone_point in rul_zone_points:
                        if zone_point not in visited_right:
                            right_main_skeleton.append(zone_point)
                            visited_right.add(zone_point)
                    
                    # If there's a single branch candidate, follow it as right main continuation              
                    if len(rul_branch_candidates) == 1:
                        exit_point = rul_branch_candidates[0]['exit_point']
                        print(f"  Following single branch from zone exit: {exit_point}")
                        
                        # Trace from exit point, but STOP at the next bifurcation
                        continuation_path = []
                        trace_current = exit_point
                        trace_visited = visited_right | rul_zone_points_set
                        
                        for trace_step in range(100):
                            continuation_path.append(trace_current)
                            trace_visited.add(trace_current)
                            
                            # Check if we hit a bifurcation - STOP HERE
                            if trace_current in bifurcations_set:
                                print(f"  Hit next bifurcation at {trace_current} after {len(continuation_path)} points")
                                break
                            
                            # Get next neighbors
                            next_neighbors = [n for n in skeleton_dict[trace_current] if n not in trace_visited]
                            if len(next_neighbors) == 0:
                                break
                            
                            # Select next point based on direction
                            if len(continuation_path) > 1:
                                prev = np.array(continuation_path[-2])
                                curr = np.array(trace_current)
                                direction = curr - prev
                                
                                best_neighbor = None
                                best_score = -np.inf
                                for n in next_neighbors:
                                    n_dir = np.array(n) - curr
                                    score = np.dot(direction, n_dir)
                                    if score > best_score:
                                        best_score = score
                                        best_neighbor = n
                                trace_current = best_neighbor
                            else:
                                trace_current = min(next_neighbors, key=lambda p: (p[2], p[1]))
                        
                        # Add continuation to right main
                        for point in continuation_path:
                            if point not in set(right_main_skeleton):
                                right_main_skeleton.append(point)
                                visited_right.add(point)
                        
                        # Add the last bifurcation to right_bifurcations list for next iteration                    
                        if len(continuation_path) > 0 and continuation_path[-1] in bifurcations_set:
                            new_bif = continuation_path[-1]
                            if new_bif not in right_bifurcations:  # Avoid duplicates
                                right_bifurcations.append(new_bif)
                                print(f"  Continued {len(continuation_path)} points to {new_bif}")
                                print(f"  Added to bifurcations list (now {len(right_bifurcations)} total)")
                        else:
                            print(f"  Continued {len(continuation_path)} points but no bifurcation found")
                            break
            
            if not rul_found:
                print("\n⚠ WARNING: Could not find valid RUL bifurcation with 2 branches!")
            else:
                # Classify as RUL (higher score) vs Intermedius (lower score)
                if len(rul_branch_candidates) >= 2:
                    rul_branch_candidates.sort(key=lambda x: x['score'], reverse=True)
                    rul_start_info = rul_branch_candidates[0]
                    intermedius_start_info = rul_branch_candidates[-1]
                    
                    # Get exit points
                    rul_exit_point = rul_start_info['exit_point']
                    int_exit_point = intermedius_start_info['exit_point']
                    
                    # Trace from exit points (this includes the full branch)
                    visited_with_rul_zone = visited_up_to_bif
                    
                    rul_skeleton = trace_from_point(rul_exit_point, visited_with_rul_zone, skeleton_dict, skeleton_coords, max_length=100)
                    
                    # Build path from bifurcation center through zone to exit point
                    rul_zone_point = rul_start_info['zone_point']
                    rul_bifurcation_to_rul_zone_point = []
                    
                    # Find path from bifurcation (zone center) to zone exit point
                    temp_current = tuple(rul_zone_center.astype(int))
                    temp_visited = set([temp_current])
                    rul_bifurcation_to_rul_zone_point.append(temp_current)
                    
                    # Simple breadth-first search to zone_point
                    for _ in range(20):  # Max 20 steps through zone
                        neighbors = [n for n in skeleton_dict[temp_current] 
                                    if n not in temp_visited and n in rul_zone_points_set]
                        if len(neighbors) == 0 or temp_current == rul_zone_point:
                            break
                        
                        # Pick neighbor closest to zone_point
                        rul_zone_point_arr = np.array(rul_zone_point)
                        best_neighbor = min(neighbors, 
                                        key=lambda n: np.sum((np.array(n) - rul_zone_point_arr)**2))
                        
                        rul_bifurcation_to_rul_zone_point.append(best_neighbor)
                        temp_visited.add(best_neighbor)
                        temp_current = best_neighbor
                    
                    # Combine: bifurcation → zone_point → exit_point → downstream
                    rul_skeleton = rul_bifurcation_to_rul_zone_point + rul_skeleton
                    # ==== END NEW SECTION ====
                    
                    print(f"  RUL traced: {len(rul_skeleton)} points")

                # Trace Intermedius and find RML/RLL bifurcation using continuation vector matching
                if intermedius_start_info:
                    visited_with_rul = visited_with_rul_zone | set(rul_skeleton)
                    
                    # Trace from exit point
                    int_exit_point = intermedius_start_info['exit_point']
                    
                    # Build path from bifurcation center through zone to exit point
                    int_zone_point = intermedius_start_info['zone_point']
                    rul_bifurcation_to_int_zone_point = []
                    
                    temp_current = tuple(rul_zone_center.astype(int))
                    temp_visited = set([temp_current]) | set(rul_bifurcation_to_rul_zone_point)
                    rul_bifurcation_to_int_zone_point.append(temp_current)
                    
                    for _ in range(20):
                        neighbors = [n for n in skeleton_dict[temp_current] 
                                    if n not in temp_visited and n in rul_zone_points_set]
                        if len(neighbors) == 0 or temp_current == int_zone_point:
                            break
                        
                        int_zone_point_arr = np.array(int_zone_point)
                        best_neighbor = min(neighbors,
                                        key=lambda n: np.sum((np.array(n) - int_zone_point_arr)**2))
                        
                        rul_bifurcation_to_int_zone_point.append(best_neighbor)
                        temp_visited.add(best_neighbor)
                        temp_current = best_neighbor

                    print(f"\n  Tracing intermedius to establish reference direction...")
                    intermedius_initial = trace_from_point(
                        int_exit_point, 
                        visited_with_rul, 
                        skeleton_dict, 
                        skeleton_coords, 
                        max_length=15,
                        allow_gap_jump=False
                    )

                    # Calculate reference direction vector using last 30 voxels of right main + ALL available intermedius
                    # Get last 30 points from right main bronchus (before RUL zone)
                    right_main_before_rul_zone = []
                    for point in reversed(right_main_skeleton):
                        if point not in rul_zone_points_set:
                            right_main_before_rul_zone.insert(0, point)
                            if len(right_main_before_rul_zone) >= 30:
                                break

                    # Combine: right main + ALL available intermedius (not just first 15)
                    if len(intermedius_initial) >= 15:
                        # Standard case: use first 15 of intermedius
                        reference_path = right_main_before_rul_zone + intermedius_initial[:15]
                        print(f"    Using first 15 voxels of intermedius")
                    elif len(intermedius_initial) >= 5:
                        # Fallback: use ALL available intermedius voxels (even if < 15)
                        reference_path = right_main_before_rul_zone + intermedius_initial
                        print(f"    ⚠ SHORT INTERMEDIUS: Using all {len(intermedius_initial)} available voxels")
                    else:
                        # Very short trace: use only right main
                        reference_path = right_main_before_rul_zone
                        print(f"    ⚠ VERY SHORT INTERMEDIUS: Using only right main skeleton")

                    if len(reference_path) >= 2:
                        reference_start = np.array(reference_path[0])
                        reference_end = np.array(reference_path[-1])
                        reference_vector = reference_end - reference_start
                        reference_vector = reference_vector / np.linalg.norm(reference_vector)  # Normalize
                        
                        print(f"    ✓ Reference vector established from {len(reference_path)} points")
                        print(f"      Total contribution: {len(right_main_before_rul_zone)} right main + {len(intermedius_initial)} intermedius")
                        print(f"    Reference direction: {reference_vector}")

                    else:
                        print(f"    ✗ ERROR: Could not build adequate reference path")
                        reference_vector = None

                    # Now trace the full intermedius, stopping when no continuation branch exists
                    if reference_vector is not None:
                        intermedius_full_path = rul_bifurcation_to_int_zone_point.copy()
                        current_point = int_exit_point
                        visited_intermedius = visited_with_rul | set(rul_bifurcation_to_int_zone_point)
                        
                        print(f"\n  Tracing intermedius with continuation vector matching...")
                        print(f"    Looking for branches aligned with reference vector...")

                        for step in range(200):
                            intermedius_full_path.append(current_point)
                            visited_intermedius.add(current_point)
                            
                            # Get unvisited neighbors
                            neighbors = skeleton_dict.get(current_point, [])
                            unvisited_neighbors = [n for n in neighbors if n not in visited_intermedius]
                            
                            if len(unvisited_neighbors) == 0:
                                print(f"    ✓ Intermedius ended at step {step}: dead end")
                                break
                            
                            # Check ALL unvisited neighbors for alignment (not just at bifurcations)
                            if len(unvisited_neighbors) >= 1:
                                continuation_found = False
                                best_continuation = None
                                best_angle = None
                                
                                # Only print detailed info at bifurcations
                                is_bifurcation = current_point in bifurcations_set and len(unvisited_neighbors) >= 2
                                
                                if is_bifurcation:
                                    print(f"\n    Testing bifurcation at step {step}, point {current_point}")
                                    print(f"      Unvisited neighbors: {len(unvisited_neighbors)}")
                                
                                for idx, neighbor in enumerate(unvisited_neighbors):
                                    if is_bifurcation:
                                        print(f"\n      Testing neighbor {idx+1}/{len(unvisited_neighbors)}: {neighbor}")
                                    
                                    # Trace lookahead from this neighbor
                                    lookahead = trace_from_point(
                                        neighbor,
                                        visited_intermedius,
                                        skeleton_dict,
                                        skeleton_coords,
                                        max_length=25,
                                        allow_gap_jump=False
                                    )
                                    
                                    if is_bifurcation:
                                        print(f"        Lookahead length: {len(lookahead)} voxels")
                                    
                                    if len(lookahead) >= 2:  # Need at least 2 points
                                        # Calculate direction vector using first 2 voxels
                                        branch_start = np.array(lookahead[0])
                                        branch_end = np.array(lookahead[1])
                                        branch_vector = branch_end - branch_start
                                        branch_vector_normalized = branch_vector / np.linalg.norm(branch_vector)
                                        
                                        # Calculate angle with reference vector
                                        cos_angle = np.dot(reference_vector, branch_vector_normalized)
                                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                                        
                                        if is_bifurcation:
                                            print(f"        ✓ Valid branch:")
                                            print(f"          Branch vector (voxels 1-2): {branch_vector}")
                                            print(f"          Angle with reference: {angle_deg:.1f}°")
                                        
                                        # Check if this path continues in the reference direction (angle < 120°)
                                        if angle_deg < 120:
                                            if is_bifurcation:
                                                print(f"          ✓✓ CONTINUATION CANDIDATE (angle < 120°)")
                                            
                                            if not continuation_found or angle_deg < best_angle:
                                                continuation_found = True
                                                best_continuation = neighbor
                                                best_angle = angle_deg
                                                
                                                if is_bifurcation:
                                                    print(f"          ✓✓✓ NEW BEST continuation (angle: {angle_deg:.1f}°)")
                                        else:
                                            if is_bifurcation:
                                                print(f"          ✗ Not a continuation (angle {angle_deg:.1f}° >= 120°)")
                                    else:
                                        if is_bifurcation:
                                            print(f"        ✗ Branch too short ({len(lookahead)} < 2 voxels)")
                                
                                # Make decision
                                if is_bifurcation:
                                    print(f"\n      DECISION:")
                                
                                if continuation_found:
                                    if is_bifurcation:
                                        print(f"        ✓ Found continuation branch: {best_continuation}")
                                        print(f"        ✓ Best angle: {best_angle:.1f}°")
                                    current_point = best_continuation
                                    continue
                                else:
                                    if is_bifurcation:
                                        print(f"        ✗ No continuation branch found (no branch < 120°)")
                                    print(f"    ✓ Intermedius ended at step {step}: no path aligned with reference (all branches > 120°)")
                                    break

                        intermedius_skeleton = intermedius_full_path
                        print(f"\n  Intermedius skeleton: {len(intermedius_skeleton)} points")

                        # ============================================
                        # VALIDATE: Check if intermedius terminal point is well-aligned with reference
                        # ============================================
                        if len(intermedius_skeleton) > 5:
                            terminal_point = np.array(intermedius_skeleton[-1])
                            origin_point = np.array(rul_bifurcation_to_int_zone_point[0])  # Use zone start as origin
                            
                            # Calculate actual path vector
                            actual_path_vector = terminal_point - origin_point
                            actual_path_vector_normalized = actual_path_vector / np.linalg.norm(actual_path_vector)
                            
                            # Calculate angle with reference vector
                            cos_angle = np.dot(reference_vector, actual_path_vector_normalized)
                            terminal_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                            
                            print(f"\n  INTERMEDIUS PATH VALIDATION:")
                            print(f"    Current path length: {len(intermedius_skeleton)} voxels")
                            print(f"    Terminal point: {tuple(terminal_point.astype(int))}")
                            print(f"    Path vector angle with reference: {terminal_angle:.1f}°")
                            
                            # ALWAYS search for longest aligned path
                            print(f"\n    Searching for LONGEST path aligned with reference vector...")
                            
                            def find_all_paths_with_scores(start_point, visited_base, max_depth=200):
                                """Find all possible paths and score them by length + alignment"""
                                path_candidates = []
                                
                                # DFS to explore all branches (DFS better for finding long paths)
                                stack = [(start_point, [start_point], visited_base | {start_point})]
                                
                                paths_explored = 0
                                while stack and paths_explored < 10000:  # Limit iterations
                                    current, path, visited_local = stack.pop()
                                    paths_explored += 1
                                    
                                    if len(path) > max_depth:
                                        continue
                                    
                                    neighbors = skeleton_dict.get(current, [])
                                    unvisited = [n for n in neighbors if n not in visited_local]
                                    
                                    if len(unvisited) == 0:
                                        # Dead end - score this path
                                        if len(path) >= 10:  # Only consider paths with at least 10 voxels
                                            term_point = np.array(current)
                                            path_vector = term_point - origin_point
                                            path_length_3d = np.linalg.norm(path_vector)
                                            path_vector_norm = path_vector / (path_length_3d + 1e-10)
                                            
                                            # Calculate alignment angle
                                            cos_ang = np.dot(reference_vector, path_vector_norm)
                                            angle = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                                            
                                            # Score: prioritize length, but penalize misalignment
                                            # Only consider paths with angle < 30° (well-aligned)
                                            if angle < 30:
                                                # Score = length * alignment_bonus
                                                alignment_bonus = (30 - angle) / 30  # 1.0 at 0°, 0.0 at 60°
                                                score = len(path) * (0.5 + 0.5 * alignment_bonus)
                                                
                                                path_candidates.append({
                                                    'path': path,
                                                    'terminal': current,
                                                    'length': len(path),
                                                    'angle': angle,
                                                    'score': score,
                                                    'length_3d': path_length_3d
                                                })
                                    else:
                                        # Continue exploring - add neighbors to stack
                                        for neighbor in unvisited:
                                            new_path = path + [neighbor]
                                            new_visited = visited_local | {neighbor}
                                            stack.append((neighbor, new_path, new_visited))
                                
                                return path_candidates
                            
                            # Find all possible paths from intermedius exit point
                            all_paths = find_all_paths_with_scores(
                                int_exit_point,
                                visited_with_rul | set(rul_bifurcation_to_int_zone_point),
                                max_depth=200
                            )
                            
                            if len(all_paths) > 0:
                                # Sort by score (length * alignment) descending
                                all_paths.sort(key=lambda x: x['score'], reverse=True)
                                
                                # Show top 5 candidates
                                print(f"\n    Top candidates:")
                                for i, candidate in enumerate(all_paths[:5]):
                                    print(f"      {i+1}. Length: {candidate['length']} voxels, "
                                        f"Angle: {candidate['angle']:.1f}°, "
                                        f"Score: {candidate['score']:.1f}")
                                
                                # Select the best one (highest score = longest + best aligned)
                                best_path = all_paths[0]
                                
                                # Only replace if significantly better (longer and well-aligned)
                                if best_path['length'] > len(intermedius_skeleton) * 0.8:  # At least 80% as long
                                    print(f"\n    ✓ Selected LONGEST aligned path:")
                                    print(f"      Length: {best_path['length']} voxels")
                                    print(f"      Terminal angle: {best_path['angle']:.1f}°")
                                    print(f"      3D length: {best_path['length_3d']:.1f} voxels")
                                    
                                    # Replace intermedius skeleton with best path
                                    intermedius_skeleton = rul_bifurcation_to_int_zone_point + best_path['path']
                                    print(f"    ✓ Updated intermedius skeleton: {len(intermedius_skeleton)} points")
                                else:
                                    print(f"    ✓ Current path is already the longest aligned path")
                            else:
                                print(f"    ⚠ No well-aligned paths found (all angles > 60°)")
                                print(f"    Keeping current path")
                        else:
                            print(f"  ⚠ Intermedius skeleton too short for validation")
                        # ============================================
                        
                        # ============================================
                        # FIND RML BIFURCATION ON EXISTING INTERMEDIUS
                        # ============================================
                        print(f"\nSearching for RML bifurcation on intermedius...")
                        print(f"  Intermedius length: {len(intermedius_skeleton)} voxels")

                        # Find all bifurcations on the EXISTING intermedius path (excluding RUL zone)
                        intermedius_bifurcations = []
                        rul_zone_radius_mm_check = 0.5 * 10

                        for point in intermedius_skeleton:
                            if point in bifurcations_set:
                                # Check if point is outside RUL zone
                                point_arr = np.array(point)
                                dist_from_rul = np.sqrt(np.sum(((point_arr - rul_zone_center) * voxel_spacing)**2))
                                
                                if dist_from_rul > rul_zone_radius_mm_check:
                                    # Also check minimum distance from RUL zone exit
                                    distance_from_int_start = np.linalg.norm(
                                        np.array(point) - np.array(int_exit_point)
                                    )
                                    if distance_from_int_start >= 2:  # At least 2 voxels from intermedius start
                                        intermedius_bifurcations.append(point)

                        print(f"  Found {len(intermedius_bifurcations)} candidate bifurcations on intermedius")

                        # ============================================
                        # SEARCH FOR VALID RML BIFURCATION
                        # ============================================
                        rml_found = False
                        rml_skeleton = None
                        rll_skeleton = None
                        rml_length_voxels = None
                        rml_length_mm = None
                        rml_terminal_point = None

                        if len(intermedius_bifurcations) > 0:
                            for bif_idx, test_rml_bif in enumerate(intermedius_bifurcations):
                                print(f"\n  Testing RML bifurcation {bif_idx+1}/{len(intermedius_bifurcations)} at: {test_rml_bif}")
                                
                                # Create 0.3cm bifurcation zone around this bifurcation
                                rml_zone_radius_cm = 0.3
                                rml_zone_radius_mm = rml_zone_radius_cm * 10
                                rml_zone_center = np.array(test_rml_bif)
                                
                                # Find all skeleton points in RML bifurcation zone
                                rml_zone_points = []
                                for coord in skeleton_coords:
                                    coord_arr = np.array(coord)
                                    dist_mm = np.sqrt(np.sum(((coord_arr - rml_zone_center) * voxel_spacing)**2))
                                    if dist_mm <= rml_zone_radius_mm:
                                        rml_zone_points.append(tuple(coord))
                                
                                rml_zone_points_set = set(rml_zone_points)
                                print(f"    RML ZONE: {len(rml_zone_points)} skeleton points")
                                
                                # Build intermedius path up to this bifurcation (from ORIGINAL intermedius)
                                intermedius_to_rml_bif = []
                                for p in intermedius_skeleton:
                                    intermedius_to_rml_bif.append(p)
                                    if p == test_rml_bif or p in rml_zone_points_set:
                                        break
                                
                                if len(intermedius_to_rml_bif) < 2:
                                    print(f"    ✗ Intermedius path too short, skipping")
                                    continue
                                
                                # Calculate mean Y of intermedius UP TO this bifurcation
                                intermedius_mean_y = np.mean([p[1] for p in intermedius_to_rml_bif])
                                print(f"    Intermedius mean Y (up to bifurcation): {intermedius_mean_y:.1f}")

                                bifurcation_y = rml_zone_center[1]  # Y coordinate of zone center (bifurcation point)
                                print(f"    Bifurcation point Y: {bifurcation_y:.1f}")
                                
                                # Get the continuation of intermedius AFTER this bifurcation
                                intermedius_continuation = []
                                found_bifurcation = False
                                for p in intermedius_skeleton:
                                    if found_bifurcation:
                                        intermedius_continuation.append(p)
                                    if p == test_rml_bif:
                                        found_bifurcation = True
                                
                                print(f"    Intermedius continuation after bifurcation: {len(intermedius_continuation)} voxels")
                                
                                # Find exit points from RML zone
                                visited_intermedius_to_bif = visited_with_rul | set(intermedius_to_rml_bif)
                                rml_exit_points = []

                                for zone_point in rml_zone_points:
                                    neighbors = skeleton_dict[zone_point]
                                    for neighbor in neighbors:
                                        if neighbor not in rml_zone_points_set and neighbor not in visited_intermedius_to_bif:
                                            rml_exit_points.append({
                                                'zone_point': zone_point,
                                                'exit_point': neighbor
                                            })

                                print(f"    Exit points: {len(rml_exit_points)}")

                                # Trace lookaheads from exit points
                                visited_up_to_rml_bif = visited_intermedius_to_bif | rml_zone_points_set
                                rml_branch_candidates = []
                                
                                print(f"\n    LOOKAHEAD TRACING (need ≥6 points for valid candidate):")
                                for i, exit_info in enumerate(rml_exit_points):

                                    lookahead_path = trace_from_point(
                                        exit_info['exit_point'],
                                        visited_up_to_rml_bif,
                                        skeleton_dict,
                                        skeleton_coords,
                                        max_length=15,
                                        allow_gap_jump=False,
                                    )
                                    
                                    print(f"        Lookahead path length: {len(lookahead_path)}")
                                    
                                    if len(lookahead_path) >= 6:
                                        # Calculate mean Y of the lookahead
                                        mean_y = np.mean([p[1] for p in lookahead_path[:-1]])
                                        
                                        print(f"        ✓ VALID - Length: {len(lookahead_path)}")
                                        print(f"          Mean Y: {mean_y:.1f}")
                                        
                                        # Check if this matches intermedius continuation
                                        matches_continuation = False

                                        # DEBUG: Show what we're comparing
                                        lookahead_set = set(lookahead_path[:-1])  # First 6 points of lookahead
                                        intermedius_full_set = set(intermedius_skeleton)  
                                        overlap = len(lookahead_set & intermedius_full_set)

                                        if overlap >= 1:  # At least 1 voxels overlap
                                            matches_continuation = True
                                            print(f"          → CONTINUATION branch ({overlap} voxels overlap with intermedius)")
                                        else:
                                            print(f"          → NEW branch ({overlap} voxels overlap)")

                                        rml_branch_candidates.append({
                                            'zone_point': exit_info['zone_point'],
                                            'exit_point': exit_info['exit_point'],
                                            'lookahead_path': lookahead_path[:-1],
                                            'mean_y': mean_y,
                                            'is_continuation': matches_continuation
                                        })
                                
                                print(f"\n    Branch candidates with valid lookahead: {len(rml_branch_candidates)}")
                                
                                # Separate continuation vs new branches
                                continuation_candidates = [c for c in rml_branch_candidates if c['is_continuation']]
                                new_branches = [c for c in rml_branch_candidates if not c['is_continuation']]
                                
                                print(f"    Continuation branches: {len(continuation_candidates)}")
                                print(f"    New branches: {len(new_branches)}")

                                # ============================================
                                # ORIENTATION-AWARE ANTERIOR DETECTION
                                # ============================================
                                # Determine Y-axis direction from affine matrix
                                y_increasing_is_anterior = affine[1, 1] > 0

                                if y_increasing_is_anterior:
                                    # Standard case: Higher Y = more anterior
                                    print(f"    Y-axis: Higher values = anterior (affine[1,1] = {affine[1,1]:.3f})")
                                    anterior_comparison = lambda mean_y, bif_y: mean_y > bif_y
                                    anterior_sort_reverse = True
                                else:
                                    # Flipped case: Lower Y = more anterior
                                    print(f"    Y-axis: Lower values = anterior (affine[1,1] = {affine[1,1]:.3f})")
                                    anterior_comparison = lambda mean_y, bif_y: mean_y < bif_y
                                    anterior_sort_reverse = False
                                # ============================================
                                
                                # We need: 1 continuation (matching intermedius) + 1 anterior branch (RML)
                                if len(continuation_candidates) >= 1 and len(new_branches) >= 1:
                                    # Find the most anterior new branch (lowest Y)
                                    new_branches.sort(key=lambda x: x['mean_y'], reverse=anterior_sort_reverse)
                                    rml_candidate = new_branches[0]

                                    # Check if it's truly anterior using orientation-aware comparison
                                    if anterior_comparison(rml_candidate['mean_y'], bifurcation_y):
                                        print(f"    ✓ Found valid RML bifurcation!")
                                        print(f"      RML branch mean Y: {rml_candidate['mean_y']:.1f} ({'anterior' if y_increasing_is_anterior else 'posterior'} to {bifurcation_y:.1f})")
                                    
                                        # Get RML exit point
                                        rml_exit_point = rml_candidate['exit_point']
                                        print(f"      RML exit point: {rml_exit_point}")
                                        
                                        # Trace RML from exit point (exclude intermedius from visited)
                                        visited_for_rml_trace = visited_up_to_rml_bif
                                        
                                        rml_downstream = trace_from_point(
                                            rml_exit_point, 
                                            visited_for_rml_trace, 
                                            skeleton_dict, 
                                            skeleton_coords, 
                                            max_length=200,
                                            allow_gap_jump=True
                                        )
                                        
                                        # Build path from bifurcation center through zone to RML exit point
                                        rml_zone_point = rml_candidate['zone_point']
                                        bifurcation_to_rml = []
                                        
                                        temp_current = tuple(rml_zone_center.astype(int))
                                        temp_visited = set([temp_current])
                                        bifurcation_to_rml.append(temp_current)
                                        
                                        for _ in range(20):
                                            neighbors = [n for n in skeleton_dict[temp_current] 
                                                        if n not in temp_visited and n in rml_zone_points_set]
                                            if len(neighbors) == 0 or temp_current == rml_zone_point:
                                                break
                                            
                                            rml_zone_point_arr = np.array(rml_zone_point)
                                            best_neighbor = min(neighbors, 
                                                            key=lambda n: np.sum((np.array(n) - rml_zone_point_arr)**2))
                                            
                                            bifurcation_to_rml.append(best_neighbor)
                                            temp_visited.add(best_neighbor)
                                            temp_current = best_neighbor
                                        
                                        # Combine: bifurcation → zone_point → exit_point → downstream
                                        rml_skeleton = bifurcation_to_rml + rml_downstream
                                        
                                        print(f"      RML traced: {len(rml_skeleton)} points")
                                        
                                        # DON'T modify intermedius_skeleton - keep it as is!
                                        print(f"      Intermedius preserved: {len(intermedius_skeleton)} points")
                                        
                                        # Store zone center for visualization
                                        int_zone_center = rml_zone_center
                                        
                                        rml_found = True

                                        # ====================================================
                                        # MEASURE RML-Bronchus LENGTH AND OTHER RML-B METRICS
                                        # ====================================================
                                        print(f"\n      Measuring RML length...")

                                        rml_terminal_point = None
                                        rml_length_voxels = None
                                        rml_length_mm = None
                                        rml_length_curved_mm = None

                                        rml_full_angle = None

                                        rml_full_sagittal_angle = None
                                        rml_full_coronal_angle = None
                                        rml_full_transverse_angle = None

                                        rml_cross_section_mean_5 = None
                                        rml_cross_section_median_5 = None
                                        rml_cross_section_min_5 = None
                                        rml_cross_section_orifice_5 = None

                                        rml_real_curved_length_mm_5 = None
                                        rml_volume_mm3_5 = None

                                        # Find first bifurcation on RML with 2+ valid branches
                                        # Search for bifurcations along RML skeleton
                                        for idx, rml_point in enumerate(rml_skeleton):
                                            if tuple(rml_point) not in bifurcations_set:
                                                continue  # Not a bifurcation
                                            
                                            # Skip if too close to RML origin (first 5 voxels)
                                            if idx < 5:
                                                continue
                                            
                                            print(f"        Testing RML bifurcation at index {idx}, point {rml_point}")
                                            
                                            # Get neighbors from this bifurcation
                                            neighbors = skeleton_dict.get(tuple(rml_point), [])
                                            
                                            # Build visited set: everything up to this point on RML + intermedius + rul
                                            visited_up_to_here = visited_up_to_rml_bif | set([tuple(p) for p in rml_skeleton[:idx+1]])
                                            
                                            unvisited_neighbors = [n for n in neighbors if n not in visited_up_to_here]
                                            
                                            if len(unvisited_neighbors) < 2:
                                                continue  # Need at least 2 branches
                                            
                                            # Trace lookaheads from each neighbor
                                            valid_branches = 0
                                            
                                            for neighbor in unvisited_neighbors:
                                                lookahead = trace_from_point(
                                                    neighbor,
                                                    visited_up_to_here,
                                                    skeleton_dict,
                                                    skeleton_coords,
                                                    max_length=5,
                                                    allow_gap_jump=False
                                                )
                                                
                                                if len(lookahead) >= 3:
                                                    valid_branches += 1
                                                    print(f"          Branch to {neighbor}: {len(lookahead)} voxels ✓")
                                            
                                            # Check if this bifurcation has 2+ valid branches
                                            if valid_branches >= 2:
                                                rml_terminal_point = tuple(rml_point)
                                                
                                                # Calculate length in voxels (from RML origin to terminal point)
                                                rml_length_voxels = idx + 1 
                                                
                                                # Calculate length in mm (Euclidean distance with voxel spacing)
                                                rml_origin = np.array(rml_skeleton[0])
                                                rml_terminal = np.array(rml_terminal_point)
                                                
                                                # Physical distance accounting for voxel spacing
                                                physical_distance = np.sqrt(np.sum(((rml_terminal - rml_origin) * voxel_spacing)**2))
                                                rml_length_mm = physical_distance
                                                
                                                # ============================================
                                                # CALCULATE CURVED LENGTH (sum of segment lengths)
                                                # ============================================
                                                rml_path_to_terminal = rml_skeleton[:idx+1]
                                                
                                                if len(rml_path_to_terminal) >= 2:
                                                    curved_length = 0.0
                                                    for i in range(len(rml_path_to_terminal) - 1):
                                                        p1 = np.array(rml_path_to_terminal[i])
                                                        p2 = np.array(rml_path_to_terminal[i + 1])
                                                        
                                                        # Physical distance between consecutive points
                                                        segment_length = np.sqrt(np.sum(((p2 - p1) * voxel_spacing)**2))
                                                        curved_length += segment_length
                                                    
                                                    
                                                    rml_length_curved_mm = curved_length
                                                    
                                                    print(f"        ✓ Found RML terminal bifurcation!")
                                                    print(f"          Terminal point: {rml_terminal_point}")
                                                    print(f"          Valid branches: {valid_branches}")
                                                    print(f"          RML straight length: {rml_length_mm:.1f} mm")
                                                    print(f"          RML curved length: {rml_length_curved_mm:.1f} mm")
                                                else:
                                                    rml_length_curved_mm = rml_length_mm  # Fallback to straight length
                                                    print(f"        ✓ Found RML terminal bifurcation!")
                                                    print(f"          Terminal point: {rml_terminal_point}")
                                                    print(f"          Valid branches: {valid_branches}")
                                                    print(f"          RML length: {rml_length_voxels} voxels = {rml_length_mm:.1f} mm")
                                                # ============================================
                                                break

                                        if rml_terminal_point is None:
                                            print(f"        ⚠ No terminal bifurcation found on RML")
                                            print(f"        Using full RML length: {len(rml_skeleton)} voxels")
                                            rml_length_voxels = len(rml_skeleton)
                                            rml_terminal_point = tuple(rml_skeleton[-1])
                                            
                                            # Calculate full length
                                            rml_origin = np.array(rml_skeleton[0])
                                            rml_terminal = np.array(rml_terminal_point)
                                            physical_distance = np.sqrt(np.sum(((rml_terminal - rml_origin) * voxel_spacing)**2))
                                            rml_length_mm = physical_distance
                                            
                                            
                                            # Calculate full CURVED length
                                            if len(rml_skeleton) >= 2:
                                                curved_length = 0.0
                                                for i in range(len(rml_skeleton) - 1):
                                                    p1 = np.array(rml_skeleton[i])
                                                    p2 = np.array(rml_skeleton[i + 1])
                                                    segment_length = np.sqrt(np.sum(((p2 - p1) * voxel_spacing)**2))
                                                    curved_length += segment_length
                                                
                                                rml_length_curved_mm = curved_length
                                                print(f"        Full RML straight length: {rml_length_mm:.1f} mm")
                                                print(f"        Full RML curved length: {rml_length_curved_mm:.1f} mm")
                                            else:
                                                rml_length_curved_mm = rml_length_mm
                                                print(f"        Full RML length: {rml_length_voxels} voxels = {rml_length_mm:.1f} mm")
                                        # ============================================

                                        # ============================================
                                        # ADDITIONAL RML MEASUREMENTS
                                        # ============================================
                                        print(f"\n      Computing additional RML measurements...")

                                        # 1. RML Full Angle (Full Vectors)
                                        # Angle between full intermedius vector and full RML vector
                                        rml_full_angle = None

                                        # Get RML path to terminal
                                        rml_path_to_terminal = []
                                        for point in rml_skeleton:
                                            rml_path_to_terminal.append(point)
                                            if tuple(point) == rml_terminal_point:
                                                break

                                        # Get intermedius path before bifurcation
                                        intermedius_before_bif = []
                                        for p in intermedius_skeleton:
                                            intermedius_before_bif.append(p)
                                            if p == test_rml_bif or p in rml_zone_points_set:
                                                break

                                        if len(intermedius_before_bif) >= 2 and len(rml_path_to_terminal) >= 2:
                                            # Intermedius direction vector (first to last point)
                                            int_start = np.array(intermedius_before_bif[0])
                                            int_end = np.array(intermedius_before_bif[-1])
                                            int_vector = (int_end - int_start) * voxel_spacing
                                            int_vector_norm = int_vector / np.linalg.norm(int_vector)
                                            
                                            # RML direction vector (first to last point)
                                            rml_start = np.array(rml_path_to_terminal[0])
                                            rml_end = np.array(rml_path_to_terminal[-1])
                                            rml_vector = (rml_end - rml_start) * voxel_spacing
                                            rml_vector_norm = rml_vector / np.linalg.norm(rml_vector)
                                            
                                            # Calculate angle between vectors
                                            cos_angle = np.dot(int_vector_norm, rml_vector_norm)
                                            rml_full_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                                        # Print all measurements
                                        print(f"\n      === RML MEASUREMENTS ===")
                                        print(f"        RML Straight Length: {rml_length_mm:.1f} mm")
                                        print(f"        RML Full Angle (Full): {rml_full_angle:.1f}°" if rml_full_angle else "        RML Full Angle: N/A")
                                        # ============================================

                                        # ============================================
                                        # RML CROSS-SECTIONAL AREA MEASUREMENTS
                                        # ============================================
                                        print(f"\n      === RML CROSS-SECTIONAL AREAS ===")
                                        print(f"      Measuring perpendicular cross-sections at every voxel along RML...")

                                        rml_cross_sections = []

                                        if len(rml_path_to_terminal) >= 3:
                                            # ==============================================
                                            # FIRST: Calculate smoothed normals for ALL points
                                            # ==============================================
                                            print(f"      Calculating smoothed normals for cross-section measurement...")
                                            
                                            smoothed_normals = []
                                            SMOOTHING_WINDOW = 5  # Use ±5 voxels for smoothing
                                            
                                            for idx in range(len(rml_path_to_terminal)):
                                                start_idx = max(0, idx - SMOOTHING_WINDOW)
                                                end_idx = min(len(rml_path_to_terminal) - 1, idx + SMOOTHING_WINDOW)
                                                
                                                # Calculate direction from START to END of window
                                                if end_idx > start_idx:
                                                    p_start = np.array(rml_path_to_terminal[start_idx])
                                                    p_end = np.array(rml_path_to_terminal[end_idx])
                                                    window_direction = (p_end - p_start) * voxel_spacing
                                                    
                                                    # Normalize
                                                    if np.linalg.norm(window_direction) > 0:
                                                        normal = window_direction / np.linalg.norm(window_direction)
                                                    else:
                                                        # Fallback
                                                        if idx < len(rml_path_to_terminal) - 1:
                                                            p1 = np.array(rml_path_to_terminal[idx])
                                                            p2 = np.array(rml_path_to_terminal[idx + 1])
                                                            direction = (p2 - p1) * voxel_spacing
                                                            normal = direction / np.linalg.norm(direction)
                                                        else:
                                                            normal = np.array([0, 0, 1])
                                                else:
                                                    normal = np.array([0, 0, 1])
                                                
                                                smoothed_normals.append(normal)
                                            
                                            print(f"      ✓ Calculated {len(smoothed_normals)} smoothed normals")
                                            
                                            # ==================================================
                                            # NOW: Measure cross-sections using smoothed normals
                                            # ==================================================
                                            rml_origin = np.array(rml_path_to_terminal[0])
                                            
                                            for i in range(len(rml_path_to_terminal)):
                                                point = rml_path_to_terminal[i]
                                                
                                                # Calculate distance from origin in mm
                                                distance_from_origin_mm = np.sqrt(np.sum(((np.array(point) - rml_origin) * voxel_spacing)**2))
                                                
                                                # USE SMOOTHED NORMAL (already normalized)
                                                direction_normalized = smoothed_normals[i]
                                                
                                                # Calculate cross-section with smoothed direction
                                                cross_section_area = calculate_perpendicular_cross_section(
                                                    point, direction_normalized, airway_mask, voxel_spacing
                                                )
                                                
                                                rml_cross_sections.append({
                                                    'index': i,
                                                    'point': point,
                                                    'area_mm2': cross_section_area,
                                                    'distance_from_origin_mm': distance_from_origin_mm,
                                                    'direction': direction_normalized  # Store smoothed direction
                                                })
                                                
                                                # Print every measurement
                                                print(f"        Voxel {i:3d} at {point} (distance: {distance_from_origin_mm:.1f} mm): {cross_section_area:.2f} mm²")
                                            
                                            # Summary statistics with DISTANCE-BASED filtering
                                            areas = [cs['area_mm2'] for cs in rml_cross_sections]
                                            distances = [cs['distance_from_origin_mm'] for cs in rml_cross_sections]
                                            
                                            # Filter: exclude first 5mm from origin only
                                            EXCLUSION_DISTANCE_MM = 5
                                            
                                            # Find indices that are >= 5mm from origin
                                            valid_indices = [
                                                i for i in range(len(rml_cross_sections))
                                                if distances[i] >= EXCLUSION_DISTANCE_MM
                                            ]
                                            
                                            if len(valid_indices) > 0:
                                                filtered_areas = [areas[i] for i in valid_indices]
                                                
                                                rml_cross_section_mean_5 = np.mean(filtered_areas)
                                                rml_cross_section_median_5 = np.median(filtered_areas)
                                                rml_cross_section_min_5 = np.min(filtered_areas)
                                                
                                                # RML ORIFICE CROSS-SECTION: First voxel at or after 5mm
                                                orifice_idx = valid_indices[0]
                                                rml_cross_section_orifice_5 = areas[orifice_idx]
                                                
                                                print(f"\n      Cross-section Summary (FILTERED - excluding first 5mm):")
                                                print(f"        Exclusion distance: {EXCLUSION_DISTANCE_MM} mm from zone center")
                                                print(f"        Number of valid measurements: {len(filtered_areas)}")
                                                print(f"        First valid voxel index: {valid_indices[0]} (at {distances[valid_indices[0]]:.1f} mm)")
                                                print(f"        Last valid voxel index: {valid_indices[-1]} (at {distances[valid_indices[-1]]:.1f} mm)")
                                                print(f"        Orifice area (first valid voxel): {rml_cross_section_orifice_5:.2f} mm²")
                                                print(f"        Mean area: {rml_cross_section_mean_5:.2f} mm²")
                                                print(f"        Median area: {rml_cross_section_median_5:.2f} mm²")
                                                print(f"        Min area: {rml_cross_section_min_5:.2f} mm²")
                                                
                                                # Store indices and areas for visualization
                                                filtered_cs_indices = valid_indices
                                                all_cross_sections = areas
                                                
                                                print(f"        Stored {len(filtered_cs_indices)} filtered indices for visualization")
                                            else:
                                                print(f"        ⚠ No measurements beyond {EXCLUSION_DISTANCE_MM} mm from origin")
                                                rml_cross_section_mean_5 = None
                                                rml_cross_section_median_5 = None
                                                rml_cross_section_min_5 = None
                                                rml_cross_section_orifice_5 = None
                                        else:
                                            print(f"        ⚠ RML path too short for cross-section measurement")
                                            rml_cross_section_mean_5 = None
                                            rml_cross_section_median_5 = None
                                            rml_cross_section_min_5 = None
                                            rml_cross_section_orifice_5 = None

                                        # ============================================
                                        # RML VOLUME (from 5mm to terminal)
                                        # ============================================
                                        print(f"\n      === RML VOLUME (from 5mm to terminal) ===")

                                        rml_volume_mm3_5 = None

                                        if len(valid_indices) > 0:
                                            # Get the same segment of the path
                                            start_idx = valid_indices[0]
                                            end_idx = valid_indices[-1]
                                            
                                            print(f"        Computing distance transforms for RML volume...")
                                            
                                            # Create skeleton mask for RML segment in measurement region
                                            rml_segment_mask = np.zeros_like(airway_mask, dtype=bool)
                                            for point in rml_path_to_terminal[start_idx:end_idx+1]:
                                                rml_segment_mask[tuple(point)] = True
                                            
                                            # Create skeleton masks for OTHER airways that could interfere with RML
                                            other_airways_mask = np.zeros_like(airway_mask, dtype=bool)
                                            
                                            # Add intermedius (main competitor for RML voxels)
                                            if intermedius_skeleton:
                                                for point in intermedius_skeleton:
                                                    other_airways_mask[tuple(point)] = True
                                            
                                            # Add RUL (nearby airway)
                                            if rul_skeleton:
                                                for point in rul_skeleton:
                                                    other_airways_mask[tuple(point)] = True
                                            
                                            # Add the parts of RML skeleton OUTSIDE the measurement region
                                            if rml_skeleton:
                                                for i, point in enumerate(rml_skeleton):
                                                    if i < start_idx or i > end_idx:
                                                        other_airways_mask[tuple(point)] = True
                                            
                                            print(f"        RML segment skeleton points: {np.sum(rml_segment_mask)}")
                                            print(f"        Other airways skeleton points: {np.sum(other_airways_mask)}")
                                            
                                            # Compute distance transforms
                                            dist_rml = ndimage.distance_transform_edt(~rml_segment_mask)
                                            dist_rml[~airway_mask] = np.inf  # Set non-airway voxels to infinity
                                            
                                            dist_other = ndimage.distance_transform_edt(~other_airways_mask)
                                            dist_other[~airway_mask] = np.inf
                                            
                                            # Assign airway voxels to RML if they're closer to RML segment than to other airways
                                            rml_volume_mask_raw = (airway_mask) & (dist_rml < dist_other)
                                            
                                            print(f"        Raw RML voxels (before filtering): {np.sum(rml_volume_mask_raw)}")
                                            
                                            # ============================================
                                            # FILTER TO LARGEST CONNECTED COMPONENT ONLY
                                            # ============================================
                                            if np.sum(rml_volume_mask_raw) > 0:
                                                # Label connected components
                                                labeled_components, num_components = ndimage.label(rml_volume_mask_raw)
                                                
                                                print(f"        Found {num_components} connected components")
                                                
                                                if num_components > 1:
                                                    # Find the component that contains the RML skeleton
                                                    # Use the middle point of the RML segment as reference
                                                    middle_idx = (start_idx + end_idx) // 2
                                                    reference_point = tuple(rml_path_to_terminal[middle_idx])
                                                    reference_label = labeled_components[reference_point]
                                                    
                                                    print(f"        Reference point (middle of RML): {reference_point}")
                                                    print(f"        Reference component label: {reference_label}")
                                                    
                                                    # Keep only the component containing the skeleton
                                                    rml_volume_mask = (labeled_components == reference_label)
                                                    
                                                    # Show sizes of all components
                                                    for comp_id in range(1, num_components + 1):
                                                        comp_size = np.sum(labeled_components == comp_id)
                                                        if comp_id == reference_label:
                                                            print(f"          Component {comp_id}: {comp_size} voxels ✓ (contains RML skeleton)")
                                                        else:
                                                            print(f"          Component {comp_id}: {comp_size} voxels ✗ (excluded)")
                                                else:
                                                    print(f"        Only 1 connected component found (no filtering needed)")
                                                    rml_volume_mask = rml_volume_mask_raw
                                            else:
                                                print(f"        ⚠ No RML voxels found!")
                                                rml_volume_mask = rml_volume_mask_raw
                                            
                                            # Calculate volume
                                            voxel_volume_mm3 = np.prod(voxel_spacing)
                                            rml_volume_mm3_5 = np.sum(rml_volume_mask) * voxel_volume_mm3
                                            
                                            print(f"        Measurement region: voxel {start_idx} (at {distances[start_idx]:.1f} mm) to voxel {end_idx} (at {distances[end_idx]:.1f} mm)")
                                            print(f"        Number of airway voxels (filtered): {np.sum(rml_volume_mask)}")
                                            print(f"        Voxel volume: {voxel_volume_mm3:.4f} mm³")
                                            print(f"        Total RML volume: {rml_volume_mm3_5:.1f} mm³ ({rml_volume_mm3_5/1000:.2f} mL)")
                                        else:
                                            print(f"        ⚠ Not enough measurements to calculate volume")


                                        # ============================================
                                        # RML PLANE ANGLE MEASUREMENTS
                                        # ============================================
                                        print(f"\n      === RML PLANE ANGLES ===")
                                        
                                        # Full RML vector
                                        rml_full_sagittal_angle = None
                                        rml_full_coronal_angle = None
                                        rml_full_transverse_angle = None
                                        
                                        if len(rml_path_to_terminal) >= 2:
                                            # Full vector: first to last point
                                            rml_full_start = np.array(rml_path_to_terminal[0])
                                            rml_full_end = np.array(rml_path_to_terminal[-1])
                                            rml_full_vector = (rml_full_end - rml_full_start) * voxel_spacing
                                            
                                            # Normalize
                                            rml_full_vector_norm = rml_full_vector / np.linalg.norm(rml_full_vector)
                                            
                                            # Calculate plane angles using arcsin of perpendicular component
                                            # Sign indicates direction
                                            
                                            # Sagittal plane (Y-Z plane): perpendicular component is X
                                            rml_full_sagittal_angle = np.degrees(np.arcsin(np.clip(rml_full_vector_norm[0], -1, 1)))
                                            
                                            # Coronal plane (X-Z plane): perpendicular component is Y  
                                            rml_full_coronal_angle = np.degrees(np.arcsin(np.clip(rml_full_vector_norm[1], -1, 1)))
                                            
                                            # Transverse plane (X-Y plane): perpendicular component is Z
                                            rml_full_transverse_angle = np.degrees(np.arcsin(np.clip(rml_full_vector_norm[2], -1, 1)))
                                            
                                            print(f"\n        Full RML (first to last):")
                                            print(f"          Sagittal plane angle: {rml_full_sagittal_angle:+.1f}° (+ = right, - = left)" if rml_full_sagittal_angle is not None else "          Sagittal plane angle: N/A")
                                            print(f"          Coronal plane angle: {rml_full_coronal_angle:+.1f}° (+ = anterior, - = posterior)" if rml_full_coronal_angle is not None else "          Coronal plane angle: N/A")
                                            print(f"          Transverse plane angle: {rml_full_transverse_angle:+.1f}° (+ = superior, - = inferior)" if rml_full_transverse_angle is not None else "          Transverse plane angle: N/A")
                                        # ============================================

                                        # ============================================
                                         # RML "REAL" LENGTH (between filtered cross-sections)
                                        # ============================================
                                        print(f"\n      === RML 'REAL' LENGTH (from 5mm to terminal) ===")

                                        rml_real_curved_length_mm_5 = None

                                        if len(valid_indices) > 0:
                                            # Get indices: first valid (at 5mm) to last (terminal)
                                            start_idx = valid_indices[0]  # First voxel at/after 5mm
                                            end_idx = valid_indices[-1]   # Last valid (terminal voxel)
                                                
                                            rml_real_path = rml_path_to_terminal[start_idx:end_idx+1]
                                                
                                            if len(rml_real_path) >= 2:
                                                # Straight length
                                                real_start = np.array(rml_real_path[0])
                                                real_end = np.array(rml_real_path[-1])
                                                rml_real_length_mm = np.sqrt(np.sum(((real_end - real_start) * voxel_spacing)**2))
                                                    
                                                # Curved length (sum of segments)
                                                curved_real_length = 0.0
                                                for i in range(len(rml_real_path) - 1):
                                                    p1 = np.array(rml_real_path[i])
                                                    p2 = np.array(rml_real_path[i + 1])
                                                    segment_length = np.sqrt(np.sum(((p2 - p1) * voxel_spacing)**2))
                                                    curved_real_length += segment_length
                                                    
                                                rml_real_curved_length_mm_5 = curved_real_length
                                                    
                                                print(f"        Measurement region: voxel {start_idx} (at {distances[start_idx]:.1f} mm) to voxel {end_idx} (at {distances[end_idx]:.1f} mm)")
                                                print(f"        Real straight length: {rml_real_length_mm:.1f} mm")
                                                print(f"        Real curved length: {rml_real_curved_length_mm_5:.1f} mm")
                                            else:
                                                print(f"        ⚠ Real path too short")
                                        else:
                                            print(f"        ⚠ Not enough measurements to calculate real length")

                                        # DON'T modify intermedius_skeleton - keep it as is!
                                        print(f"      Intermedius preserved: {len(intermedius_skeleton)} points")

                                        break
                                    else:
                                        print(f"    ✗ New branch not anterior (mean_y {rml_candidate['mean_y']:.1f} >= bifurcation Y {bifurcation_y:.1f})")
                                        print(f"    → This is likely a posterior branch, continuing search...")
                                else:
                                    print(f"    ✗ Insufficient branches (need 1 continuation + 1 new anterior)")
                                    print(f"    → Continuing to next bifurcation...")

                        else:
                            print(f"  ⚠ No bifurcations found on intermedius for RML detection")

                        if not rml_found:
                            print(f"\n  ⚠ WARNING: Could not find valid RML bifurcation")
                        else:
                            print(f"\n  ✓ RML IDENTIFIED")
                            print(f"    Intermedius: {len(intermedius_skeleton)} points (UNCHANGED)")
                            print(f"    RML begins at bifurcation: {rml_skeleton[0] if rml_skeleton else None}")

    print("\n✓ Analysis complete!")
    print(f"  Trachea: {len(trachea_skeleton)} points")
    print(f"  Right main: {len(right_main_skeleton) if right_main_skeleton else 0} points")
    print(f"  RUL: {len(rul_skeleton) if rul_skeleton else 0} points")
    print(f"  Intermedius: {len(intermedius_skeleton) if intermedius_skeleton else 0} points")

    # ============================================
    # TRACE LEFT MAIN BRONCHUS AND FIND LUL/LLL BIFURCATION
    # ============================================
    left_main_skeleton = None
    lul_skeleton = None
    lll_skeleton = None
    left_bifurcations = []
    lul_zone_center = None

    if left_main_start:
        print(f"\nTracing left main bronchus from {left_main_start}...")
        left_main_skeleton = []
        current = left_main_start
        visited_left = visited_trachea | carinal_zone_points_set | {current}
        left_main_skeleton.append(current)
        left_bifurcations = []

        MIN_LEFT_MAIN_LENGTH = 10  # voxels
        
        for step in range(100):
            neighbors = [n for n in skeleton_dict[current] if n not in visited_left]

            if len(neighbors) == 0:
                if step < 20:
                    print(f"    >> No neighbors available - STOPPING")
                break
            
            # Check if current point is a bifurcation
            if current in bifurcations_set:
                # Calculate distance from left main start
                distance_from_start = np.linalg.norm(
                    np.array(current) - np.array(left_main_start)
                )
                
                if distance_from_start < MIN_LEFT_MAIN_LENGTH:
                    print(f"    >> Found bifurcation at {current} but too close to carina ({distance_from_start:.1f} voxels)")
                    print(f"    >> Continuing through it...")
                else:
                    # Far enough from carina - this is a valid LUL/LLL bifurcation
                    left_bifurcations.append(current)
                    print(f"    >> HIT VALID BIFURCATION - STOPPING at {current} (distance: {distance_from_start:.1f} voxels)")
                    break
            
            # Continue tracing
            if len(left_main_skeleton) > 1:
                prev = np.array(left_main_skeleton[-2])
                curr = np.array(current)
                direction = curr - prev
                
                if step < 20:
                    print(f"    Using direction from {tuple(prev.astype(int))} to {current}")
                    print(f"    Direction vector: {direction}")
                
                best_neighbor = None
                best_score = -np.inf
                
                for n in neighbors:
                    n_dir = np.array(n) - curr
                    
                    # Base score: dot product with previous direction
                    score = np.dot(direction, n_dir)
                    
                    # DIRECTIONAL BIAS: Prefer left (decreasing X) and down (decreasing Z)
                    # Add bonus for moving left (negative X direction)
                    if n_dir[0] > 0:  # Moving right in image (left anatomically)
                        score += abs(n_dir[0]) * 2.0
                    elif n_dir[0] < 0:  # Moving left in image (right anatomically)
                        score -= abs(n_dir[0]) * 3.0

                    # NEW: Prefer posterior (decreasing Y)
                    if n_dir[1] < 0:  # Moving posterior
                        score += abs(n_dir[1]) * 2.5  # Strong bonus
                    elif n_dir[1] > 0:  # Moving anterior
                        score -= abs(n_dir[1]) * 2.0  # Penalty

                    # Prefer moving down (decreasing Z)
                    if n_dir[2] < 0:  # Moving inferior
                        score += abs(n_dir[2]) * 0.5
                    
                    if step < 20:
                        print(f"      Neighbor {n}: dir={n_dir}, score={score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_neighbor = n
                
                next_point = best_neighbor
                if step < 20:
                    print(f"    >> SELECTED: {next_point} (score: {best_score:.2f})")
            else:
                next_point = neighbors[0]
                if step < 20:
                    print(f"    >> SELECTED (first step): {next_point}")
            
            left_main_skeleton.append(next_point)
            visited_left.add(next_point)
            current = next_point
        
        # Continue with LUL/LLL analysis
        if len(left_bifurcations) > 0:
            lul_found = False
            
            for bif_idx, test_left_bif in enumerate(left_bifurcations):
                print(f"\nTesting LUL/LLL bifurcation {bif_idx+1}/{len(left_bifurcations)} at: {test_left_bif}")
                
                # Create 0.5cm bifurcation zone around this left bifurcation
                lul_zone_radius_cm = 0.5
                lul_zone_radius_mm = lul_zone_radius_cm * 10
                lul_zone_center = np.array(test_left_bif)
                
                # Find all skeleton points in LUL bifurcation zone
                lul_zone_points = []
                for coord in skeleton_coords:
                    coord_arr = np.array(coord)
                    dist_mm = np.sqrt(np.sum(((coord_arr - lul_zone_center) * voxel_spacing)**2))
                    if dist_mm <= lul_zone_radius_mm:
                        lul_zone_points.append(tuple(coord))
                
                lul_zone_points_set = set(lul_zone_points)
                print(f"  LUL/LLL ZONE: {len(lul_zone_points)} skeleton points")
                
                # Find exit points from LUL zone
                visited_left = visited_trachea | carinal_zone_points_set | set(left_main_skeleton)
                lul_exit_points = []
                
                for zone_point in lul_zone_points:
                    neighbors = skeleton_dict[zone_point]
                    for neighbor in neighbors:
                        if neighbor not in lul_zone_points_set and neighbor not in visited_left:
                            lul_exit_points.append({
                                'zone_point': zone_point,
                                'exit_point': neighbor
                            })
                
                print(f"  Exit points: {len(lul_exit_points)}")
                
                # Trace lookaheads from exit points
                visited_up_to_bif = visited_left | lul_zone_points_set
                lul_branch_candidates = []
                
                print(f"\n  LOOKAHEAD TRACING (need ≥10 points for valid candidate):")
                for i, exit_info in enumerate(lul_exit_points):
                    print(f"    Exit {i+1} from {exit_info['zone_point']} → {exit_info['exit_point']}:")
                    
                    lookahead_path = trace_from_point(
                        exit_info['exit_point'],
                        visited_up_to_bif,
                        skeleton_dict,
                        skeleton_coords,
                        max_length=15,
                        allow_gap_jump=False
                    )
                    
                    print(f"      Lookahead path length: {len(lookahead_path)}")

                    # Trace the FULL branch to see total available length
                    full_branch = trace_from_point(
                        exit_info['exit_point'],
                        visited_up_to_bif,
                        skeleton_dict,
                        skeleton_coords,
                        max_length=200
                    )
                    print(f"      Full branch length: {len(full_branch)} voxels")
                    
                    if len(lookahead_path) >= 10:
                        lookahead_point = lookahead_path[9]  # 10th point (index 9)
                        direction = np.array(lookahead_point) - lul_zone_center
                        score = direction[2] + direction[1]  # Z + Y score
                        
                        print(f"      ✓ VALID - Length: {len(lookahead_path)}")
                        print(f"        10th point: {lookahead_point}")
                        print(f"        Direction from zone center: {direction}")
                        print(f"        Score (Z+Y): {score:.1f}")
                        
                        lul_branch_candidates.append({
                            'zone_point': exit_info['zone_point'],
                            'exit_point': exit_info['exit_point'],
                            'lookahead_path': lookahead_path[:10],
                            'score': score
                        })
                    else:
                        print(f"      ✗ INVALID - Only {len(lookahead_path)} points (need ≥10)")
                
                print(f"\n  Branch candidates with valid lookahead: {len(lul_branch_candidates)}")
                
                # Check if we have at least 2 candidates
                if len(lul_branch_candidates) >= 2:
                    print(f"  ✓ Found valid LUL/LLL bifurcation with {len(lul_branch_candidates)} branches!")
                    lul_found = True
                    break
                else:
                    print(f"  ✗ Insufficient branches, continuing to next bifurcation...")
                    
                    # Mark all zone points as part of left main bronchus
                    print(f"  Marking {len(lul_zone_points)} zone points as left main")
                    for zone_point in lul_zone_points:
                        if zone_point not in visited_left:
                            left_main_skeleton.append(zone_point)
                            visited_left.add(zone_point)
                    
                    # If there's a single branch candidate, follow it as left main continuation
                    if len(lul_branch_candidates) == 1:
                        exit_point = lul_branch_candidates[0]['exit_point']
                        print(f"  Following single branch from zone exit: {exit_point}")
                        
                        # Trace from exit point, but STOP at the next bifurcation
                        continuation_path = []
                        trace_current = exit_point
                        trace_visited = visited_left | lul_zone_points_set
                        
                        for trace_step in range(100):
                            continuation_path.append(trace_current)
                            trace_visited.add(trace_current)
                            
                            # Check if we hit a bifurcation - STOP HERE
                            if trace_current in bifurcations_set:
                                print(f"  Hit next bifurcation at {trace_current} after {len(continuation_path)} points")
                                break
                            
                            # Get next neighbors
                            next_neighbors = [n for n in skeleton_dict[trace_current] if n not in trace_visited]
                            if len(next_neighbors) == 0:
                                break
                            
                            # Select next point based on direction
                            if len(continuation_path) > 1:
                                prev = np.array(continuation_path[-2])
                                curr = np.array(trace_current)
                                direction = curr - prev
                                
                                best_neighbor = None
                                best_score = -np.inf
                                for n in next_neighbors:
                                    n_dir = np.array(n) - curr
                                    score = np.dot(direction, n_dir)
                                    if score > best_score:
                                        best_score = score
                                        best_neighbor = n
                                trace_current = best_neighbor
                            else:
                                trace_current = min(next_neighbors, key=lambda p: (p[2], p[1]))
                        
                        # Add continuation to left main
                        for point in continuation_path:
                            if point not in set(left_main_skeleton):
                                left_main_skeleton.append(point)
                                visited_left.add(point)
                        
                        # Add the last bifurcation to left_bifurcations list for next iteration
                        if len(continuation_path) > 0 and continuation_path[-1] in bifurcations_set:
                            new_bif = continuation_path[-1]
                            if new_bif not in left_bifurcations:
                                left_bifurcations.append(new_bif)
                                print(f"  Continued {len(continuation_path)} points to {new_bif}")
                                print(f"  Added to bifurcations list (now {len(left_bifurcations)} total)")
                        else:
                            print(f"  Continued {len(continuation_path)} points but no bifurcation found")
                            break
            
            if not lul_found:
                print("\n⚠ WARNING: Could not find valid LUL/LLL bifurcation with 2 branches!")
            else:
                # Classify as LUL (higher score) vs LLL (lower score)
                if len(lul_branch_candidates) >= 2:
                    lul_branch_candidates.sort(key=lambda x: x['score'], reverse=True)
                    lul_start_info = lul_branch_candidates[0]
                    lll_start_info = lul_branch_candidates[-1]
                    
                    print(f"  LUL branch score: {lul_start_info['score']:.1f}")
                    print(f"  LLL branch score: {lll_start_info['score']:.1f}")
                    
                    # Get exit points
                    lul_exit_point = lul_start_info['exit_point']
                    lll_exit_point = lll_start_info['exit_point']
                    
                    print(f"  LUL exit point: {lul_exit_point}")
                    print(f"  LLL exit point: {lll_exit_point}")
                    
                    # Trace from exit points
                    visited_with_lul_zone = visited_up_to_bif
                    
                    lul_skeleton = trace_from_point(lul_exit_point, visited_with_lul_zone, skeleton_dict, skeleton_coords, max_length=100)
                    
                    # Build path from bifurcation center through zone to LUL exit point
                    lul_zone_point = lul_start_info['zone_point']
                    bifurcation_to_lul_zone = []
                    
                    temp_current = tuple(lul_zone_center.astype(int))
                    temp_visited = set([temp_current])
                    bifurcation_to_lul_zone.append(temp_current)
                    
                    for _ in range(20):
                        neighbors = [n for n in skeleton_dict[temp_current] 
                                    if n not in temp_visited and n in lul_zone_points_set]
                        if len(neighbors) == 0 or temp_current == lul_zone_point:
                            break
                        
                        lul_zone_point_arr = np.array(lul_zone_point)
                        best_neighbor = min(neighbors, 
                                        key=lambda n: np.sum((np.array(n) - lul_zone_point_arr)**2))
                        
                        bifurcation_to_lul_zone.append(best_neighbor)
                        temp_visited.add(best_neighbor)
                        temp_current = best_neighbor
                    
                    lul_skeleton = bifurcation_to_lul_zone + lul_skeleton
                    print(f"  LUL traced: {len(lul_skeleton)} points")
                    
                    # Trace LLL
                    visited_with_lul = visited_with_lul_zone | set(lul_skeleton)
                    
                    lll_skeleton = trace_from_point(lll_exit_point, visited_with_lul, skeleton_dict, skeleton_coords, max_length=100)
                    
                    # Build path from bifurcation center through zone to LLL exit point
                    lll_zone_point = lll_start_info['zone_point']
                    bifurcation_to_lll_zone = []
                    
                    temp_current = tuple(lul_zone_center.astype(int))
                    temp_visited = set([temp_current]) | set(bifurcation_to_lul_zone)
                    bifurcation_to_lll_zone.append(temp_current)
                    
                    for _ in range(20):
                        neighbors = [n for n in skeleton_dict[temp_current] 
                                    if n not in temp_visited and n in lul_zone_points_set]
                        if len(neighbors) == 0 or temp_current == lll_zone_point:
                            break
                        
                        lll_zone_point_arr = np.array(lll_zone_point)
                        best_neighbor = min(neighbors,
                                        key=lambda n: np.sum((np.array(n) - lll_zone_point_arr)**2))
                        
                        bifurcation_to_lll_zone.append(best_neighbor)
                        temp_visited.add(best_neighbor)
                        temp_current = best_neighbor
                    
                    lll_skeleton = bifurcation_to_lll_zone + lll_skeleton
                    print(f"  LLL traced: {len(lll_skeleton)} points")

    print("\n✓ Analysis complete!")
    print(f"  Trachea: {len(trachea_skeleton)} points")
    print(f"  Right main: {len(right_main_skeleton) if right_main_skeleton else 0} points")
    print(f"  RUL: {len(rul_skeleton) if rul_skeleton else 0} points")
    print(f"  Intermedius: {len(intermedius_skeleton) if intermedius_skeleton else 0} points")
    print(f"  RML: {len(rml_skeleton) if rml_skeleton else 0} points")
    print(f"  Left main: {len(left_main_skeleton) if left_main_skeleton else 0} points")
    print(f"  LUL: {len(lul_skeleton) if lul_skeleton else 0} points")
    print(f"  LLL: {len(lll_skeleton) if lll_skeleton else 0} points")


    def trace_all_branches_recursive(start_point, visited_set, skeleton_dict, skeleton_coords, 
                                    stop_zones=None, max_depth=1000):
        """
        Recursively trace ALL branches from a starting point using depth-first search.
        
        Parameters:
        -----------
        start_point : tuple
            Starting point coordinates
        visited_set : set
            Set of already visited points (will be modified in-place)
        skeleton_dict : dict
            Dictionary of skeleton connectivity
        skeleton_coords : ndarray
            Array of all skeleton coordinates
        stop_zones : list of dict, optional
            List of zone dictionaries with 'center' and 'radius_mm' to stop at zone boundaries
            Example: [{'center': np.array([x,y,z]), 'radius_mm': 5.0}, ...]
        max_depth : int
            Maximum recursion depth to prevent infinite loops
        
        Returns:
        --------
        set : All points reached from start_point
        """
        if stop_zones is None:
            stop_zones = []
        
        all_points = set()
        
        def dfs_explore(current_point, depth=0):
            """Recursive DFS helper function"""
            if depth > max_depth:
                return
            
            # Add current point
            all_points.add(current_point)
            visited_set.add(current_point)
            
            # Get neighbors
            neighbors = skeleton_dict.get(current_point, [])
            
            for neighbor in neighbors:
                # Skip if already visited
                if neighbor in visited_set:
                    continue
                
                # Check if neighbor is in any stop zone
                in_stop_zone = False
                if stop_zones:
                    neighbor_arr = np.array(neighbor)
                    for zone in stop_zones:
                        zone_center = zone['center']
                        zone_radius = zone['radius_mm']
                        
                        # Calculate distance with voxel spacing (assuming global access)
                        dist_mm = np.sqrt(np.sum(((neighbor_arr - zone_center) * voxel_spacing)**2))
                        
                        if dist_mm <= zone_radius:
                            in_stop_zone = True
                            break
                
                # If in stop zone, don't explore further
                if in_stop_zone:
                    continue
                
                # Recursively explore this neighbor
                dfs_explore(neighbor, depth + 1)
        
        # Start the DFS
        dfs_explore(start_point, 0)
        
        return all_points
    


    if visualize:
        # Use off-screen rendering if screenshot_path is provided
        if screenshot_path:
            plotter = pv.Plotter(off_screen=True)
            plotter.window_title = f"CT Airway Analysis – MRN {Path(nii_file_path).stem.replace('.nii','')}"
        else:
            plotter = pv.Plotter()
            plotter.window_title = f"CT Airway Analysis – MRN {Path(nii_file_path).stem.replace('.nii','')}"
        
        # Transform ALL coordinates for visualization (voxel → mm + RAS orientation)
        skeleton_coords_viz = transform_coords_for_visualization(skeleton_coords, voxel_spacing, affine)

        # ============================================
        # SAVE ORIGINAL SKELETONS (Before extension for visualization)
        # ============================================
        print("\n=== SAVING ORIGINAL SKELETONS FOR MEASUREMENT ===")

        trachea_skeleton_original = trachea_skeleton.copy() if trachea_skeleton else None
        right_main_skeleton_original = right_main_skeleton.copy() if right_main_skeleton else None
        left_main_skeleton_original = left_main_skeleton.copy() if left_main_skeleton else None
        intermedius_skeleton_original = intermedius_skeleton.copy() if intermedius_skeleton else None

        print(f"  Saved original skeletons:")
        print(f"    Trachea: {len(trachea_skeleton_original) if trachea_skeleton_original else 0} points")
        print(f"    Right Main: {len(right_main_skeleton_original) if right_main_skeleton_original else 0} points")
        print(f"    Left Main: {len(left_main_skeleton_original) if left_main_skeleton_original else 0} points")
        print(f"    Intermedius: {len(intermedius_skeleton_original) if intermedius_skeleton_original else 0} points")

        
        # ============================================
        # EXTEND RIGHT AND LEFT MAIN BRONCHI TO CARINAL ZONE CENTER
        # ============================================
        print("\n=== EXTENDING MAIN BRONCHI TO CARINAL ZONE CENTER ===")

        if carina_found and carinal_zone_center is not None:
            # Extend RIGHT MAIN BRONCHUS - include all carinal zone points on right side (X < carina X)
            if right_main_skeleton and right_main_start:
                print("\nExtending right main bronchus to carinal zone center...")
                
                carinal_zone_right_points = [
                    p for p in carinal_zone_points_set
                    if p[0] <= carinal_zone_center[0]  # Right side (X ≤ center X)
                ]
                
                # Add these to right main (avoiding duplicates)
                existing_right_main = set(right_main_skeleton)
                new_points = [p for p in carinal_zone_right_points if p not in existing_right_main]
                
                right_main_skeleton = new_points + right_main_skeleton
                print(f"  ✓ Extended right main by {len(new_points)} points from carinal zone")
            
            # Similar for LEFT MAIN
            if left_main_skeleton and left_main_start:
                print("\nExtending left main bronchus to carinal zone center...")
                
                carinal_zone_left_points = [
                    p for p in carinal_zone_points_set
                    if p[0] >= carinal_zone_center[0]  # Left side (X ≥ center X)
                ]
                
                existing_left_main = set(left_main_skeleton)
                new_points = [p for p in carinal_zone_left_points if p not in existing_left_main]
                
                left_main_skeleton = new_points + left_main_skeleton
                print(f"  ✓ Extended left main by {len(new_points)} points from carinal zone")

        print("\n✓ Main bronchi extension complete!")

        # ============================================
        # EXTEND TRACHEA INTO EXCLUSION ZONE
        # ============================================
        print("\n=== EXTENDING TRACHEA INTO EXCLUSION ZONE ===")

        if trachea_skeleton and len(trachea_skeleton) > 0 and carinal_zone_center is not None:
            print("\nExtending trachea into superior exclusion zone...")
            
            # DEBUG: Show all trachea points and their Z coordinates
            trachea_z_coords = [p[2] for p in trachea_skeleton]
            print(f"  Current trachea has {len(trachea_skeleton)} points")
            print(f"  Trachea Z range: {min(trachea_z_coords)} to {max(trachea_z_coords)}")
            print(f"  Carinal zone center: {carinal_zone_center}")
            print(f"  Carinal zone radius: {zone_radius_mm} mm")
            
            # Find all skeleton points OUTSIDE carinal zone that are:
            # (a) connected to trachea or carinal zone, AND
            # (b) SUPERIOR to the carinal zone center (higher Z)
            print(f"  Finding SUPERIOR skeleton points outside carinal zone...")
            
            carinal_zone_center_z = carinal_zone_center[2]
            
            starting_points = []
            for coord in skeleton_coords:
                coord_tuple = tuple(coord)
                
                # Skip if already in trachea
                if coord_tuple in set(trachea_skeleton):
                    continue
                
                # Skip if inside carinal zone
                if coord_tuple in carinal_zone_points_set:
                    continue
                
                # CRITICAL: Only consider points ABOVE (superior to) carinal zone center
                if coord[2] <= carinal_zone_center_z:
                    continue
                
                # Check if connected to trachea or carinal zone
                neighbors = skeleton_dict.get(coord_tuple, [])
                connected_to_trachea = any(
                    n in set(trachea_skeleton) or n in carinal_zone_points_set
                    for n in neighbors
                )
                
                if connected_to_trachea:
                    starting_points.append(coord_tuple)
            
            print(f"  Found {len(starting_points)} superior starting points outside zone")
            
            # RECURSIVELY explore from these starting points
            trachea_extension_set = set()
            visited_for_extension = set(trachea_skeleton) | carinal_zone_points_set
            
            def explore_trachea_recursively(current_point, depth=0):
                """Recursively explore skeleton points, STOPPING when entering carinal zone"""
                if depth > 300:  # Safety limit
                    return
                
                neighbors = skeleton_dict.get(current_point, [])
                
                for neighbor in neighbors:
                    if neighbor in visited_for_extension:
                        continue
                    
                    # Skip if in carinal zone
                    if neighbor in carinal_zone_points_set:
                        visited_for_extension.add(neighbor)
                        continue
                    
                    # Add this neighbor (it's outside carinal zone)
                    trachea_extension_set.add(neighbor)
                    visited_for_extension.add(neighbor)
                    
                    # Recursively explore from this neighbor
                    explore_trachea_recursively(neighbor, depth + 1)
            
            # Explore from all superior starting points
            for start_point in starting_points:
                if start_point not in visited_for_extension:
                    trachea_extension_set.add(start_point)
                    visited_for_extension.add(start_point)
                    explore_trachea_recursively(start_point)
            
            # Add to trachea skeleton
            trachea_extension_list = list(trachea_extension_set)
            trachea_skeleton.extend(trachea_extension_list)
            
            print(f"  ✓ Extended trachea by {len(trachea_extension_list)} points (stopped at carinal zone)")
            if len(trachea_skeleton) > 0:
                new_z_range = [p[2] for p in trachea_skeleton]
                print(f"  New trachea Z range: {min(new_z_range)} to {max(new_z_range)}")

        print("\n✓ Trachea extension complete!")

        # ============================================
        # BUILD SETS FOR ANATOMICAL CLASSIFICATION
        # ============================================
        print("\n=== BUILDING ANATOMICAL CLASSIFICATION ===")

        # Convert all paths to sets for easy lookup
        trachea_set = set([tuple(p) for p in trachea_skeleton]) if trachea_skeleton else set()
        right_main_set = set([tuple(p) for p in right_main_skeleton]) if right_main_skeleton else set()
        left_main_set = set([tuple(p) for p in left_main_skeleton]) if left_main_skeleton else set()  # ← ADD THIS
        rul_skeleton_set = set([tuple(p) for p in rul_skeleton]) if rul_skeleton else set()
        rml_skeleton_set = set([tuple(p) for p in rml_skeleton]) if rml_skeleton else set()

        # Initialize sets for different regions
        rul_distal_set = set()  # Everything distal to RUL bifurcation in RUL branch
        rml_distal_set = set()  # Everything distal to RML bifurcation zone (all branches from RML)
        rll_distal_set = set()  # Everything distal to RML bifurcation zone (NOT in RML branch)
        intermedius_side_branches_set = set()  # Side branches from intermedius (between RUL and RML zones)
        intermedius_main_set = set()  # Main intermedius path (between RUL and RML zones)

        # Define stop zones
        stop_zones_list = []

        # Add RUL zone as stop zone (if it exists)
        if rul_zone_center is not None:
            stop_zones_list.append({
                'center': rul_zone_center,
                'radius_mm': 0.5 * 10,
                'name': 'RUL Zone'
            })

        # Add RML zone as stop zone (if it exists)
        if rml_found and int_zone_center is not None:
            stop_zones_list.append({
                'center': int_zone_center,
                'radius_mm': 0.3 * 10,
                'name': 'RML Zone'
            })

        print(f"\nDefined {len(stop_zones_list)} stop zones:")
        for zone in stop_zones_list:
            print(f"   - {zone['name']}: center={zone['center']}, radius={zone['radius_mm']}mm")

        # ============================================
        # 1. RUL DISTAL BRANCHES (beyond RUL bifurcation zone)
        # ============================================
        if rul_skeleton and rul_zone_center is not None:
            print("\n1. Tracing RUL distal branches (recursive DFS)...")
            rul_zone_radius_mm = 0.5 * 10
            
            # Find where RUL exits its bifurcation zone
            rul_exit_points = []
            for i, point in enumerate(rul_skeleton):
                point_arr = np.array(point)
                dist_from_rul = np.sqrt(np.sum(((point_arr - rul_zone_center) * voxel_spacing)**2))
                
                if dist_from_rul > rul_zone_radius_mm:
                    rul_exit_points.append((i, tuple(point)))
            
            if rul_exit_points:
                # Get first exit point
                first_exit_idx, first_exit_point = rul_exit_points[0]
                print(f"   RUL exits zone at index {first_exit_idx}: {first_exit_point}")
                
                # Build visited set: trachea + right main + RUL zone portion
                visited_for_rul_distal = trachea_set | right_main_set | set([tuple(p) for p in rul_skeleton[:first_exit_idx+1]])
                
                # NO stop zones for RUL - we want to explore everything distal
                rul_distal_set = trace_all_branches_recursive(
                    first_exit_point,
                    visited_for_rul_distal.copy(),  # Pass a copy so we don't modify original
                    skeleton_dict,
                    skeleton_coords,
                    stop_zones=None,  # No stop zones - explore everything
                    max_depth=1000
                )
                
                print(f"   ✓ RUL distal: {len(rul_distal_set)} points")

        # ============================================
        # 2. INTERMEDIUS: Separate main path from side branches
        # ============================================
        if intermedius_skeleton and rml_found and int_zone_center is not None:
            print("\n2. Separating intermedius main path from side branches...")
            
            # Find where intermedius path enters RML bifurcation zone
            rml_zone_radius_mm = 0.3 * 10
            
            intermedius_to_rml_zone = []
            for point in intermedius_skeleton:
                point_arr = np.array(point)
                dist_from_rml_zone = np.sqrt(np.sum(((point_arr - int_zone_center) * voxel_spacing)**2))
                
                intermedius_to_rml_zone.append(tuple(point))
                
                # Stop when entering RML zone
                if dist_from_rml_zone <= rml_zone_radius_mm:
                    break
            
            intermedius_main_set = set(intermedius_to_rml_zone)
            print(f"   ✓ Intermedius main path: {len(intermedius_main_set)} points")
            
            # Find side branches: trace from each bifurcation on intermedius main path
            print(f"   Searching for side branches on intermedius main path...")
            
            # Build base visited set
            base_visited = trachea_set | right_main_set | rul_skeleton_set | rul_distal_set | intermedius_main_set
            
            # Define stop zone: RML zone (don't explore into RML zone)
            intermedius_stop_zones = [
                {'center': int_zone_center, 'radius_mm': rml_zone_radius_mm, 'name': 'RML Zone'}
            ]
            
            side_branch_count = 0
            for point in intermedius_to_rml_zone:
                # Check if this is a bifurcation
                if point not in bifurcations_set:
                    continue
                
                # Get neighbors
                neighbors = skeleton_dict.get(point, [])
                
                # Find neighbors that are NOT in the main intermedius path
                for neighbor in neighbors:
                    if neighbor in intermedius_main_set:
                        continue  # This is the main path direction
                    if neighbor in base_visited:
                        continue  # Already visited
                    
                    # This is a side branch - trace it recursively
                    print(f"      Found side branch from {point} → {neighbor}")
                    
                    side_branch_points = trace_all_branches_recursive(
                        neighbor,
                        base_visited.copy(),
                        skeleton_dict,
                        skeleton_coords,
                        stop_zones=intermedius_stop_zones,  # Stop at RML zone
                        max_depth=1000
                    )
                    
                    # Add to intermedius side branches set
                    intermedius_side_branches_set.update(side_branch_points)
                    base_visited.update(side_branch_points)  # Update base to avoid double-counting
                    side_branch_count += 1
            
            print(f"   ✓ Found {side_branch_count} side branches")
            print(f"   ✓ Intermedius side branches (→ RLL): {len(intermedius_side_branches_set)} points")

        # ============================================
        # 3. RML DISTAL BRANCHES (beyond RML bifurcation zone - all RML branches)
        # ============================================
        if rml_skeleton and int_zone_center is not None:
            print("\n3. Tracing RML distal branches (recursive DFS)...")
            rml_zone_radius_mm = 0.3 * 10
            
            # Find where RML exits its bifurcation zone
            rml_exit_points = []
            for i, point in enumerate(rml_skeleton):
                point_arr = np.array(point)
                dist_from_rml_zone = np.sqrt(np.sum(((point_arr - int_zone_center) * voxel_spacing)**2))
                
                if dist_from_rml_zone > rml_zone_radius_mm:
                    rml_exit_points.append((i, tuple(point)))
            
            if rml_exit_points:
                first_exit_idx, first_exit_point = rml_exit_points[0]
                print(f"   RML exits zone at index {first_exit_idx}: {first_exit_point}")
                
                # Build visited set: everything EXCEPT what we want to trace as RML distal
                visited_for_rml_distal = (trachea_set | right_main_set | rul_skeleton_set | rul_distal_set |
                                        intermedius_main_set | intermedius_side_branches_set |
                                        set([tuple(p) for p in rml_skeleton[:first_exit_idx+1]]))
                
                # NO stop zones for RML - explore everything distal
                rml_distal_set = trace_all_branches_recursive(
                    first_exit_point,
                    visited_for_rml_distal.copy(),
                    skeleton_dict,
                    skeleton_coords,
                    stop_zones=None,  # No stop zones
                    max_depth=1000
                )
                
                print(f"   ✓ RML distal: {len(rml_distal_set)} points")

        # ============================================
        # 4. RLL DISTAL BRANCHES (beyond RML bifurcation zone - NOT in RML branch)
        # ============================================
        if int_zone_center is not None and rml_found:
            print("\n4. Tracing RLL distal branches (recursive DFS)...")
            rml_zone_radius_mm = 0.3 * 10
            
            # Find all skeleton points in RML bifurcation zone
            rml_zone_points_for_rll = []
            for coord in skeleton_coords:
                coord_arr = np.array(coord)
                dist_mm = np.sqrt(np.sum(((coord_arr - int_zone_center) * voxel_spacing)**2))
                if dist_mm <= rml_zone_radius_mm:
                    rml_zone_points_for_rll.append(tuple(coord))
            
            rml_zone_set = set(rml_zone_points_for_rll)
            
            # Find exit points from RML zone
            visited_up_to_rml_zone = (trachea_set | right_main_set | rul_skeleton_set | rul_distal_set |
                                    intermedius_main_set | intermedius_side_branches_set | rml_zone_set)
            
            rll_exit_points = []
            for zone_point in rml_zone_points_for_rll:
                neighbors = skeleton_dict.get(zone_point, [])
                for neighbor in neighbors:
                    if (neighbor not in rml_zone_set and 
                        neighbor not in visited_up_to_rml_zone and
                        neighbor not in rml_skeleton_set and  # NOT in RML branch
                        neighbor not in rml_distal_set):  # NOT in RML distal
                        rll_exit_points.append(neighbor)
            
            print(f"   Found {len(rll_exit_points)} potential RLL exit points from zone")
            
            # Trace from each exit point
            for idx, exit_point in enumerate(rll_exit_points):
                visited_for_rll = (visited_up_to_rml_zone | rml_skeleton_set | 
                                rml_distal_set | rll_distal_set)
                
                print(f"      Tracing RLL branch {idx+1}/{len(rll_exit_points)} from {exit_point}...")
                
                rll_branch_points = trace_all_branches_recursive(
                    exit_point,
                    visited_for_rll.copy(),
                    skeleton_dict,
                    skeleton_coords,
                    stop_zones=None,  # No stop zones
                    max_depth=1000
                )
                
                # Add to RLL distal set
                rll_distal_set.update(rll_branch_points)
                print(f"         → Found {len(rll_branch_points)} points")
            
            print(f"   ✓ RLL distal total: {len(rll_distal_set)} points")

        # ============================================
        # 5. LEFT LUNG CLASSIFICATION (LUL and LLL distal branches)
        # ============================================
        lul_skeleton_set = set([tuple(p) for p in lul_skeleton]) if lul_skeleton else set()
        lll_skeleton_set = set([tuple(p) for p in lll_skeleton]) if lll_skeleton else set()
        left_main_set = set([tuple(p) for p in left_main_skeleton]) if left_main_skeleton else set()

        lul_distal_set = set()
        lll_distal_set = set()

        # Trace LUL distal branches
        if lul_skeleton and lul_zone_center is not None:
            print("\n5. Tracing LUL distal branches (recursive DFS)...")
            lul_zone_radius_mm = 0.5 * 10
            
            # Find where LUL exits its bifurcation zone
            lul_exit_points = []
            for i, point in enumerate(lul_skeleton):
                point_arr = np.array(point)
                dist_from_lul = np.sqrt(np.sum(((point_arr - lul_zone_center) * voxel_spacing)**2))
                
                if dist_from_lul > lul_zone_radius_mm:
                    lul_exit_points.append((i, tuple(point)))
            
            if lul_exit_points:
                first_exit_idx, first_exit_point = lul_exit_points[0]
                print(f"   LUL exits zone at index {first_exit_idx}: {first_exit_point}")
                
                # Build visited set: trachea + left main + LUL zone portion
                visited_for_lul_distal = trachea_set | left_main_set | set([tuple(p) for p in lul_skeleton[:first_exit_idx+1]])
                
                lul_distal_set = trace_all_branches_recursive(
                    first_exit_point,
                    visited_for_lul_distal.copy(),
                    skeleton_dict,
                    skeleton_coords,
                    stop_zones=None,
                    max_depth=1000
                )
                
                print(f"   ✓ LUL distal: {len(lul_distal_set)} points")

        # Trace LLL distal branches
        if lll_skeleton and lul_zone_center is not None:
            print("\n6. Tracing LLL distal branches (recursive DFS)...")
            lul_zone_radius_mm = 0.5 * 10
            
            # Find where LLL exits its bifurcation zone
            lll_exit_points = []
            for i, point in enumerate(lll_skeleton):
                point_arr = np.array(point)
                dist_from_lul_zone = np.sqrt(np.sum(((point_arr - lul_zone_center) * voxel_spacing)**2))
                
                if dist_from_lul_zone > lul_zone_radius_mm:
                    lll_exit_points.append((i, tuple(point)))
            
            if lll_exit_points:
                first_exit_idx, first_exit_point = lll_exit_points[0]
                print(f"   LLL exits zone at index {first_exit_idx}: {first_exit_point}")
                
                # Build visited set: trachea + left main + LUL skeleton + LLL zone portion
                visited_for_lll_distal = (trachea_set | left_main_set | lul_skeleton_set | lul_distal_set |
                                        set([tuple(p) for p in lll_skeleton[:first_exit_idx+1]]))
                
                lll_distal_set = trace_all_branches_recursive(
                    first_exit_point,
                    visited_for_lll_distal.copy(),
                    skeleton_dict,
                    skeleton_coords,
                    stop_zones=None,
                    max_depth=1000
                )
                
                print(f"   ✓ LLL distal: {len(lll_distal_set)} points")
                
        
        # ============================================
        # VISUALIZE FULL SKELETON WITH COLOR CLASSIFICATION
        # ============================================
        print("\n=== VISUALIZING CLASSIFIED SKELETON ===")
        
        # Build colored skeleton
        skeleton_colors = []
        skeleton_points_classified = []
        
        for coord in skeleton_coords:
            coord_tuple = tuple(coord)
            
            # Classify each point
            if coord_tuple in trachea_set:
                color = [0, 255, 255]  # Cyan - Trachea
            elif coord_tuple in right_main_set:
                color = [255, 140, 0]  # Dark Orange - Right Main
            elif coord_tuple in left_main_set:
                color = [255, 0, 0]  # Red - Left Main
            elif coord_tuple in rul_skeleton_set:
                color = [0, 250, 154]  # Medium Spring Green - RUL (zone portion)
            elif coord_tuple in rul_distal_set:
                color = [0, 250, 154]  # Medium Spring Green - RUL (distal branches)
            elif coord_tuple in lul_skeleton_set:
                color = [135, 206, 250]  # Light Sky Blue - LUL (zone portion)
            elif coord_tuple in lul_distal_set:
                color = [135, 206, 250]  # Light Sky Blue - LUL (distal branches)
            elif coord_tuple in intermedius_main_set:
                color = [70, 130, 180] # Steel Blue - Intermedius main
            elif coord_tuple in intermedius_side_branches_set:
                color = [0, 128, 0]  # Green - RLL (intermedius side branches)
            elif coord_tuple in rml_skeleton_set:
                color = [255, 0, 255]  # Magenta - RML (zone portion)
            elif coord_tuple in rml_distal_set:
                color = [255, 0, 255]  # Magenta - RML (distal branches)
            elif coord_tuple in rll_distal_set:
                color = [0, 128, 0]  # Green - RLL (distal branches)
            elif coord_tuple in lll_skeleton_set:
                color = [0, 0, 128]  # Navy Blue - LLL
            elif coord_tuple in lll_distal_set:
                color = [0, 0, 128]  # Navy Blue - LLL
            else:
                color = [200, 200, 200]  # Light gray - Unclassified
            
            skeleton_points_classified.append(coord)
            skeleton_colors.append(color)
        
        # Transform to visualization space
        skeleton_points_viz = transform_coords_for_visualization(
            np.array(skeleton_points_classified), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
        )
        
        # Create colored point cloud
        skeleton_cloud = pv.PolyData(skeleton_points_viz.astype(np.float32))
        skeleton_cloud['colors'] = np.array(skeleton_colors, dtype=np.uint8)
        
        plotter.add_mesh(
            skeleton_cloud,
            scalars='colors',
            rgb=True,
            point_size=8,
            render_points_as_spheres=True,
            label='Classified Skeleton'
        )
        
        print(f"\n✓ Skeleton classification complete:")
        print(f"   Trachea: {len(trachea_set)} points")
        print(f"   Right Main: {len(right_main_set)} points")
        print(f"   Left Main: {len(left_main_set)} points")
        print(f"   RUL: {len(rul_distal_set)} points")
        print(f"   LUL: {len(lul_distal_set)} points")
        print(f"   Intermedius (main): {len(intermedius_main_set)} points")
        print(f"   Intermedius (side branches → RLL): {len(intermedius_side_branches_set)} points")
        print(f"   RML: {len(rml_distal_set)} points")
        print(f"   RLL: {len(rll_distal_set)} points")
        print(f"   LLL: {len(lll_distal_set)} points")
        
        # ============================================
        # DRAW BIFURCATION ZONE SPHERES
        # ============================================
        if carina_found and carinal_zone_center is not None:
            zone_center_viz = transform_coords_for_visualization(
                np.array([carinal_zone_center]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
            )[0]
            
            sphere = pv.Sphere(radius=zone_radius_mm, center=zone_center_viz)
            plotter.add_mesh(sphere, color='lightblue', opacity=0.3, label=f'Carinal Zone ({carina_zone_radius_cm}cm)')

        if len(right_bifurcations) > 0 and rul_zone_center is not None:
            rul_zone_center_viz = transform_coords_for_visualization(
                np.array([rul_zone_center]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
            )[0]
            
            rul_zone_radius_mm = 0.5 * 10
            rul_sphere = pv.Sphere(radius=rul_zone_radius_mm, center=rul_zone_center_viz)
            plotter.add_mesh(rul_sphere, color='lightblue', opacity=0.3, label=f'RUL Zone (0.5cm)')
        
        if rml_found and int_zone_center is not None:
            int_zone_center_viz = transform_coords_for_visualization(
                np.array([int_zone_center]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
            )[0]
            
            int_zone_radius_mm = 0.3 * 10
            int_sphere = pv.Sphere(radius=int_zone_radius_mm, center=int_zone_center_viz)
            plotter.add_mesh(int_sphere, color='lightblue', opacity=0.3, label=f'RML/RLL Zone (0.3cm)')

        # ============================================
        # DRAW RML MEASURED SEGMENT AND TERMINAL POINT
        # ============================================
        if rml_skeleton and rml_terminal_point is not None:
            measured_rml_path = []
            for point in rml_skeleton:
                measured_rml_path.append(point)
                if tuple(point) == rml_terminal_point:
                    break
            
            if len(measured_rml_path) >= 2:
                measured_rml_viz = transform_coords_for_visualization(
                    np.array(measured_rml_path), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )
                
                # Create tube for measured segment
                def path_to_polydata(path, tube_radius=0.5):
                    if path is None or len(path) < 2:
                        return None
                    points = np.array(path)
                    n_points = len(points)
                    lines = np.full((n_points - 1, 3), 2, dtype=np.int_)
                    lines[:, 1] = np.arange(n_points - 1)
                    lines[:, 2] = np.arange(1, n_points)
                    poly = pv.PolyData(points)
                    poly.lines = lines.ravel()
                    return poly.tube(radius=tube_radius)
                
                measured_rml_poly = path_to_polydata(measured_rml_viz, tube_radius=2.2)
                if measured_rml_poly:
                    plotter.add_mesh(measured_rml_poly, color='gold', label=f'RML Measured ({rml_length_mm:.1f}mm)')
                
                # Add sphere at terminal point
                terminal_viz = transform_coords_for_visualization(
                    np.array([rml_terminal_point]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )
                terminal_sphere = pv.PolyData(terminal_viz)
                plotter.add_mesh(terminal_sphere, color='gold', point_size=25,
                            render_points_as_spheres=True, label='RML Terminal Point')
        

        # ============================================
        # VISUALIZE ALL CROSS-SECTIONS AS COLORED DISCS
        # ============================================

        if 'rml_cross_sections' in locals() and len(rml_cross_sections) > 0:
            print("\n=== VISUALIZING ALL RML CROSS-SECTIONS ===")
            print(f"  Total cross-sections: {len(rml_cross_sections)}")
            
            # Find min and max areas for color mapping - ONLY FROM FILTERED INDICES
            if 'filtered_cs_indices' in locals() and len(filtered_cs_indices) > 0:
                # Get areas ONLY from filtered cross-sections
                filtered_areas = [rml_cross_sections[idx]['area_mm2'] for idx in filtered_cs_indices]
                min_area = min(filtered_areas)
                max_area = max(filtered_areas)
                print(f"  Area range (filtered only): {min_area:.2f} - {max_area:.2f} mm²")
            else:
                # Fallback: use all areas
                all_areas = [cs['area_mm2'] for cs in rml_cross_sections]
                min_area = min(all_areas)
                max_area = max(all_areas)
                print(f"  Area range (all): {min_area:.2f} - {max_area:.2f} mm²")
            
            # Color map: blue (small) → red (large)
            import matplotlib.pyplot as plt
            from matplotlib import cm
            cmap = cm.get_cmap('jet')
            
            # DEBUG: Check filtered indices
            if 'filtered_cs_indices' in locals():
                print(f"\n  DEBUG: filtered_cs_indices: {filtered_cs_indices[:10] if len(filtered_cs_indices) > 10 else filtered_cs_indices}")
                print(f"  DEBUG: First filtered idx: {filtered_cs_indices[0]}")
                print(f"  DEBUG: Last filtered idx: {filtered_cs_indices[-1]}")
            
            # Visualize using stored directions from measurement
            for cs_data in rml_cross_sections:
                idx = cs_data['index']
                
                # SKIP if not in filtered indices
                if 'filtered_cs_indices' in locals() and idx not in filtered_cs_indices:
                    continue
                
                point = cs_data['point']
                area = cs_data['area_mm2']
                
                # Get the tangent direction
                tangent = cs_data['direction']

                # Calculate radius from area
                radius = np.sqrt(area / np.pi)

                # Transform point to visualization space
                point_viz = transform_coords_for_visualization(
                    np.array([point]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]

                # ============================================
                # MANUAL DISC CREATION (bypasses pv.Disc issues)
                # ============================================
                # Create two orthonormal vectors perpendicular to tangent

                # Find the axis most different from tangent
                abs_components = np.abs(tangent)
                if abs_components[0] <= abs_components[1] and abs_components[0] <= abs_components[2]:
                    arbitrary = np.array([1.0, 0.0, 0.0])
                elif abs_components[1] <= abs_components[2]:
                    arbitrary = np.array([0.0, 1.0, 0.0])
                else:
                    arbitrary = np.array([0.0, 0.0, 1.0])

                # Create orthonormal basis perpendicular to tangent (in voxel space)
                u = np.cross(tangent, arbitrary)
                u = u / np.linalg.norm(u)
                v = np.cross(tangent, u)
                v = v / np.linalg.norm(v)

                # Transform u and v to visualization space (rotation only)
                rotation_matrix = affine[:3, :3]
                u_viz = rotation_matrix @ u
                u_viz = u_viz / np.linalg.norm(u_viz)
                v_viz = rotation_matrix @ v
                v_viz = v_viz / np.linalg.norm(v_viz)

                # Create disc points manually
                n_points = 30
                theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
                disc_points = []

                for t in theta:
                    # Point on circle in the perpendicular plane
                    p = point_viz + radius * (np.cos(t) * u_viz + np.sin(t) * v_viz)
                    disc_points.append(p)

                disc_points = np.array(disc_points)

                # Create triangular faces from center to circle
                faces = []
                n = len(disc_points)
                for i in range(n):
                    next_i = (i + 1) % n
                    faces.append([3, i, next_i, n])  # Triangle: edge_start, edge_end, center

                # Combine circle and center point
                all_points = np.vstack([disc_points, [point_viz]])

                # Create PolyData
                disc = pv.PolyData(all_points, faces=faces)

                # DEBUG
                if idx < 10 or idx == filtered_cs_indices[0] or idx == filtered_cs_indices[-1]:
                    print(f"\n  CS {idx}: point={point}, area={area:.2f} mm²")
                    print(f"    radius={radius:.2f} mm")
                    print(f"    Tangent: {tangent}")
                    print(f"    u (perp1): {u}")
                    print(f"    v (perp2): {v}")

                # Color based on area
                normalized_area = (area - min_area) / (max_area - min_area) if max_area > min_area else 0.5
                color_rgba = cmap(normalized_area)
                color_rgb = [int(c * 255) for c in color_rgba[:3]]

                opacity = 0.9

                # Add to plotter
                plotter.add_mesh(disc, color=color_rgb, opacity=opacity, show_edges=True)
            
            print(f"\n  ✓ Added {len(filtered_cs_indices) if 'filtered_cs_indices' in locals() else len(rml_cross_sections)} cross-section discs")


        # ============================================
        # VISUALIZE INTERMEDIUS AND RML VECTORS WITH ANGLE
        # ============================================
        if 'rml_full_angle' in locals() and rml_full_angle is not None and intermedius_skeleton_original:
            print("\n=== VISUALIZING INTERMEDIUS AND RML VECTORS ===")
            
            # Reconstruct rml_path_to_terminal using rml_skeleton and rml_terminal_point
            rml_path_to_terminal = []
            if rml_skeleton and 'rml_terminal_point' in locals() and rml_terminal_point is not None:
                for point in rml_skeleton:
                    rml_path_to_terminal.append(point)
                    if tuple(point) == rml_terminal_point:
                        break
            
            print(f"  RML path to terminal: {len(rml_path_to_terminal)} points")
            
            # Get the intermedius vector (ONLY UP TO RML BIFURCATION)
            intermedius_start = np.array(intermedius_skeleton_original[0])

            # Find where RML bifurcation is in the original skeleton
            rml_bif_point = test_rml_bif  # or rml_skeleton[0] if test_rml_bif not available
            intermedius_end_idx = None
            for idx, p in enumerate(intermedius_skeleton_original):
                if p == rml_bif_point or p in rml_zone_points_set:
                    intermedius_end_idx = idx
                    break

            if intermedius_end_idx is not None:
                intermedius_end = np.array(intermedius_skeleton_original[intermedius_end_idx])
            else:
                # Fallback to last point
                intermedius_end = np.array(intermedius_skeleton_original[-1])

            intermedius_vector = (intermedius_end - intermedius_start) * voxel_spacing
            
            # Get the RML vector (from bifurcation to terminal)
            if len(rml_path_to_terminal) >= 2:
                rml_start = np.array(rml_path_to_terminal[0])
                rml_end = np.array(rml_path_to_terminal[-1])
                rml_vector = (rml_end - rml_start) * voxel_spacing
                
                print(f"  RML start: {rml_start}")
                print(f"  RML end (terminal): {rml_end}")
                print(f"  RML vector length: {np.linalg.norm(rml_vector):.1f} mm")
                
                # Transform points to visualization space
                intermedius_start_viz = transform_coords_for_visualization(
                    np.array([intermedius_start]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]
                intermedius_end_viz = transform_coords_for_visualization(
                    np.array([intermedius_end]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]
                rml_start_viz = transform_coords_for_visualization(
                    np.array([rml_start]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]
                rml_end_viz = transform_coords_for_visualization(
                    np.array([rml_end]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]
                
                print(f"  Intermedius vector viz: start={intermedius_start_viz}, end={intermedius_end_viz}")
                print(f"  RML vector viz: start={rml_start_viz}, end={rml_end_viz}")
                
                # Create arrow for intermedius vector
                intermedius_arrow = pv.Arrow(
                    start=intermedius_start_viz,
                    direction=(intermedius_end_viz - intermedius_start_viz) * 2.5,
                    scale='auto',
                    tip_length=0.2,
                    tip_radius=0.1,
                    shaft_radius=0.04
                )
                plotter.add_mesh(intermedius_arrow, color='blue', opacity=0.8, label='Intermedius Vector')
                
                # Create arrow for RML vector
                rml_arrow = pv.Arrow(
                    start=rml_start_viz,
                    direction=(rml_end_viz - rml_start_viz) * 1.5,
                    scale='auto',
                    tip_length=0.2,
                    tip_radius=0.1,
                    shaft_radius=0.04
                )
                plotter.add_mesh(rml_arrow, color='red', opacity=0.8, label='RML Vector')
                
                # Add text annotation showing the angle
                angle_text = f"RML Angle: {rml_full_angle:.1f}°"
                plotter.add_point_labels(
                    [rml_start_viz + 5],
                    [angle_text],
                    font_size=20,
                    text_color='yellow',
                    point_size=0,
                    bold=True
                )
                
                print(f"  ✓ Added intermedius vector: {np.linalg.norm(intermedius_vector):.1f} mm")
                print(f"  ✓ Added RML vector: {np.linalg.norm(rml_vector):.1f} mm")
                print(f"  ✓ Angle between vectors: {rml_full_angle:.1f}°")
            else:
                print(f"  ⚠ Could not reconstruct RML path (only {len(rml_path_to_terminal)} points)")


        # ============================================
        # VISUALIZE RML VECTOR WITH ANATOMICAL PLANES
        # ============================================
        if 'rml_full_angle' in locals() and rml_full_angle is not None and rml_skeleton:
            print("\n=== VISUALIZING RML VECTOR AND ANATOMICAL PLANES ===")
            
            # Reconstruct rml_path_to_terminal using rml_skeleton and rml_terminal_point
            rml_path_to_terminal = []
            if rml_skeleton and 'rml_terminal_point' in locals() and rml_terminal_point is not None:
                for point in rml_skeleton:
                    rml_path_to_terminal.append(point)
                    if tuple(point) == rml_terminal_point:
                        break
            
            print(f"  RML path to terminal: {len(rml_path_to_terminal)} points")
            
            # Get the RML vector (from bifurcation to terminal)
            if len(rml_path_to_terminal) >= 2:
                rml_start = np.array(rml_path_to_terminal[0])
                rml_end = np.array(rml_path_to_terminal[-1])
                rml_vector = (rml_end - rml_start) * voxel_spacing
                
                print(f"  RML start: {rml_start}")
                print(f"  RML end (terminal): {rml_end}")
                print(f"  RML vector length: {np.linalg.norm(rml_vector):.1f} mm")
                
                # Transform points to visualization space
                rml_start_viz = transform_coords_for_visualization(
                    np.array([rml_start]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]
                rml_end_viz = transform_coords_for_visualization(
                    np.array([rml_end]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
                )[0]
                
                print(f"  RML vector viz: start={rml_start_viz}, end={rml_end_viz}")
                
                # Create arrow for RML vector
                rml_arrow = pv.Arrow(
                    start=rml_start_viz,
                    direction=rml_end_viz - rml_start_viz,
                    scale='auto',
                    tip_length=0.2,
                    tip_radius=0.1,
                    shaft_radius=0.04
                )
                plotter.add_mesh(rml_arrow, color='red', opacity=0.8, label='RML Vector')
                
                # ============================================
                # ADD ANATOMICAL PLANES AT RML BIFURCATION
                # ============================================
                if 'rml_full_sagittal_angle' in locals() and rml_full_sagittal_angle is not None:
                    print(f"\n  Adding anatomical planes at RML bifurcation...")
                    
                    # Plane size
                    plane_size = 30  # mm
                    
                    # Center at RML bifurcation
                    plane_center = rml_start_viz
                    
                    # Create three orthogonal planes
                    # Sagittal plane (Y-Z plane, perpendicular to X)
                    sagittal_plane = pv.Plane(
                        center=plane_center,
                        direction=(1, 0, 0),  # Normal in X direction
                        i_size=plane_size,
                        j_size=plane_size
                    )
                    plotter.add_mesh(sagittal_plane, color='cyan', opacity=0.3, label='Sagittal Plane')
                    
                    # Coronal plane (X-Z plane, perpendicular to Y)
                    coronal_plane = pv.Plane(
                        center=plane_center,
                        direction=(0, 1, 0),  # Normal in Y direction
                        i_size=plane_size,
                        j_size=plane_size
                    )
                    plotter.add_mesh(coronal_plane, color='yellow', opacity=0.3, label='Coronal Plane')
                    
                    # Transverse plane (X-Y plane, perpendicular to Z)
                    transverse_plane = pv.Plane(
                        center=plane_center,
                        direction=(0, 0, 1),  # Normal in Z direction
                        i_size=plane_size,
                        j_size=plane_size
                    )
                    plotter.add_mesh(transverse_plane, color='magenta', opacity=0.3, label='Transverse Plane')
                    
                    # Add angle labels
                    label_offset = 15  # Offset to move labels away from intersection

                    # Sagittal angle 
                    sagittal_text = f"Sagittal angle: {rml_full_sagittal_angle:+.1f}°"
                    plotter.add_point_labels(
                        [rml_start_viz + np.array([0, label_offset, 10])], 
                        [sagittal_text],
                        font_size=16,
                        text_color='cyan',
                        point_size=0,
                        bold=True
                    )

                    # Coronal angle - 
                    coronal_text = f"Coronal angle: {rml_full_coronal_angle:+.1f}°"
                    plotter.add_point_labels(
                        [rml_start_viz + np.array([label_offset, 0, 10])], 
                        [coronal_text],
                        font_size=16,
                        text_color='yellow',
                        point_size=0,
                        bold=True
                    )

                    # Transverse angle - 
                    transverse_text = f"Transverse angle: {rml_full_transverse_angle:+.1f}°"
                    plotter.add_point_labels(
                        [rml_start_viz + np.array([-5, -label_offset, 0])],  
                        [transverse_text],
                        font_size=16,
                        text_color='magenta',
                        point_size=0,
                        bold=True
                    )
                    
                    print(f"  ✓ Added anatomical planes and angle labels")
                    print(f"    Sagittal angle: {rml_full_sagittal_angle:+.1f}°")
                    print(f"    Coronal angle: {rml_full_coronal_angle:+.1f}°")
                    print(f"    Transverse angle: {rml_full_transverse_angle:+.1f}°")
                
                print(f"  ✓ Added RML vector: {np.linalg.norm(rml_vector):.1f} mm")
            else:
                print(f"  ⚠ Could not reconstruct RML path (only {len(rml_path_to_terminal)} points)")
        
        

        # ============================================
        # LEFT ZONE SPHERE
        # ============================================

        if lul_zone_center is not None:
            lul_zone_center_viz = transform_coords_for_visualization(
                np.array([lul_zone_center]), voxel_spacing, affine, skeleton_coords_reference=skeleton_coords
            )[0]
            
            lul_zone_radius_mm = 0.5 * 10
            lul_sphere = pv.Sphere(radius=lul_zone_radius_mm, center=lul_zone_center_viz)
            plotter.add_mesh(lul_sphere, color='lightblue', opacity=0.3, label=f'LUL/LLL Zone (0.5cm)')

        
        # Configure plotter
        #plotter.add_axes()
        #plotter.add_legend()
        plotter.show_grid()
        #plotter.add_text(f'Airways', position='upper_edge', font_size=12, color='black')
        
        skeleton_center_viz = skeleton_coords_viz.mean(axis=0)

        camera_position = [
            skeleton_center_viz[0],
            skeleton_center_viz[1] + 100,
            skeleton_center_viz[2] - 100
        ]

        plotter.camera.position = camera_position
        plotter.camera.focal_point = skeleton_center_viz
        plotter.camera.up = [0, 0, 1]
        plotter.reset_camera()

        #mrn_text = f"MRN: {Path(nii_file_path).stem.replace('.nii','')}"
        #plotter.add_text(mrn_text, position=(20, 20), font_size=14)
        
        if screenshot_path:
            plotter.screenshot(screenshot_path, window_size=[1920, 1080])
            plotter.close()
            print(f"  Screenshot saved: {screenshot_path}")
        else:
            plotter.show()
        
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("FINAL MEASUREMENTS SUMMARY")
    print("="*60)
        
    print("\nAIRWAY STRUCTURES:")
    print(f"  RUL: {len(rul_distal_set) if 'rul_distal_set' in locals() else 0} points")
    print(f"  LUL: {len(lul_distal_set) if 'lul_distal_set' in locals() else 0} points")
    print(f"  Intermedius: {len(intermedius_main_set) if 'intermedius_main_set' in locals() and intermedius_main_set else 0} points")
    print(f"  RML: {len(rml_distal_set) if 'rml_distal_set' in locals() else 0} points")
    print(f"  RLL: {len(rll_distal_set) if 'rll_distal_set' in locals() else 0} points")
    print(f"  LLL: {len(lll_distal_set) if 'lll_distal_set' in locals() else 0} points")
        
    if 'rml_length_mm' in locals() and rml_length_mm is not None:

        print("\n  RML ANGLES:")
        print(f"    Full angle: {rml_full_angle:.1f}°" if 'rml_full_angle' in locals() and rml_full_angle else "    Full angle: N/A")
            
        print("\n  RML PLANE ANGLES:")
        print(f"    Sagittal: {rml_full_sagittal_angle:.1f}°" if 'rml_full_sagittal_angle' in locals() and rml_full_sagittal_angle else "    Sagittal: N/A")
        print(f"    Coronal: {rml_full_coronal_angle:.1f}°" if 'rml_full_coronal_angle' in locals() and rml_full_coronal_angle else "    Coronal: N/A")
        print(f"    Transverse: {rml_full_transverse_angle:.1f}°" if 'rml_full_transverse_angle' in locals() and rml_full_transverse_angle else "    Transverse: N/A")
            
        print("\n  RML CROSS-SECTIONS:")
        print(f"    Orifice (6th voxel): {rml_cross_section_orifice_5:.2f} mm²" if 'rml_cross_section_orifice_5' in locals() and rml_cross_section_orifice_5 else "    Orifice: N/A")
        print(f"    Mean: {rml_cross_section_mean_5:.2f} mm²" if 'rml_cross_section_mean_5' in locals() and rml_cross_section_mean_5 else "    Mean: N/A")
        print(f"    Median: {rml_cross_section_median_5:.2f} mm²" if 'rml_cross_section_median_5' in locals() and rml_cross_section_median_5 else "    Median: N/A")
        print(f"    Min: {rml_cross_section_min_5:.2f} mm²" if 'rml_cross_section_min_5' in locals() and rml_cross_section_min_5 else "    Min: N/A")

        print("\n  RML 'REAL' MEASUREMENTS (between filtered cross-sections):")
        print(f"    Real length (curved): {rml_real_curved_length_mm_5:.1f} mm" if 'rml_real_curved_length_mm_5' in locals() and rml_real_curved_length_mm_5 else "    Real length (curved): N/A")
        print(f"    Volume: {rml_volume_mm3_5:.1f} mm³ ({rml_volume_mm3_5/1000:.2f} mL)" if 'rml_volume_mm3_5' in locals() and rml_volume_mm3_5 else "    Volume: N/A")
            
    print("="*60)    
        
    # Return results
    return {
        # ============================================
        # AIRWAY STRUCTURE SKELETONS 
        # ============================================
        'trachea': trachea_skeleton,
        'right_main': right_main_skeleton,
        'left_main': left_main_skeleton,
        'rul': rul_skeleton,
        'lul': lul_skeleton,
        'intermedius': intermedius_skeleton,
        'rml': rml_skeleton,
        'rll': rll_skeleton,
        'lll': lll_skeleton,
        
        # ============================================
        # AIRWAY STRUCTURE SETS (for classification/visualization)
        # ============================================
        'trachea_set': trachea_set if 'trachea_set' in locals() else None,
        'right_main_set': right_main_set if 'right_main_set' in locals() else None,
        'left_main_set': left_main_set if 'left_main_set' in locals() else None,
        'rul_set': rul_distal_set if 'rul_distal_set' in locals() else None,
        'lul_set': lul_distal_set if 'lul_distal_set' in locals() else None,
        'intermedius_main_set': intermedius_main_set if 'intermedius_main_set' in locals() else None,
        'intermedius_side_branches_set': intermedius_side_branches_set if 'intermedius_side_branches_set' in locals() else None,
        'rml_set': rml_distal_set if 'rml_distal_set' in locals() else None,
        'rll_distal_set': rll_distal_set if 'rll_distal_set' in locals() else None,
        'lll_set': lll_distal_set if 'lll_distal_set' in locals() else None,

        'carinal_zone_center': carinal_zone_center if 'carinal_zone_center' in locals() else None,
        'carina_zone_radius_cm': carina_zone_radius_cm if 'carina_zone_radius_cm' in locals() else 1.0,
        
        # ============================================
        # RML MEASUREMENTS
        # ============================================
       
        # RML Angles
        'rml_full_angle': rml_full_angle if 'rml_full_angle' in locals() else None,
        
        # RML Plane Angles
        'rml_full_sagittal_angle': rml_full_sagittal_angle if 'rml_full_sagittal_angle' in locals() else None,
        'rml_full_coronal_angle': rml_full_coronal_angle if 'rml_full_coronal_angle' in locals() else None,
        'rml_full_transverse_angle': rml_full_transverse_angle if 'rml_full_transverse_angle' in locals() else None,
        
        # RML Cross-sections
        'rml_cross_section_orifice': rml_cross_section_orifice_5 if 'rml_cross_section_orifice_5' in locals() else None,
        'rml_cross_section_mean': rml_cross_section_mean_5 if 'rml_cross_section_mean_5' in locals() else None,
        'rml_cross_section_median': rml_cross_section_median_5 if 'rml_cross_section_median_5' in locals() else None,
        'rml_cross_section_min': rml_cross_section_min_5 if 'rml_cross_section_min_5' in locals() else None,

        # RML "Real" measurements (between filtered cross-sections)
        'rml_real_curved_length_mm_5': rml_real_curved_length_mm_5 if 'rml_real_curved_length_mm_5' in locals() else None,
        'rml_volume_mm3_5': rml_volume_mm3_5 if 'rml_volume_mm3_5' in locals() else None,
    }

if __name__ == "__main__":

    nii_file_path = r"path/to/airway_segmentation.nii.gz"
    
    mrn = Path(nii_file_path).stem.replace('.nii', '')

    # Run the analysis (single file with interactive window)
    result = analyze_right_lung_airways(
        nii_file_path,
        carina_zone_radius_cm=1,
        lookahead_distance=3,
        visualize=True,
        screenshot_path=None,  # None = interactive window
        title=f"CT Right Airways 6m - MRN {mrn}"
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - Interactive window should be open")
    print("="*60)