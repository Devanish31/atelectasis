import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path
import sys

# Import the skeleton analysis from Script 1
sys.path.insert(0, str(Path(__file__).parent))
from analyze_airway_skeleton import analyze_right_lung_airways


def segment_airway_volumes(nii_file_path, skeleton_result, output_path=None):
    """
    Segment the full airway volume into anatomical regions using skeleton classification.
    
    Parameters:
    -----------
    nii_file_path : str
        Path to the airway segmentation NIfTI file
    skeleton_result : dict
        Result dictionary from analyze_right_lung_airways()
    output_path : str, optional
        Path to save the labeled volume
    
    Returns:
    --------
    labeled_volume : ndarray
        3D array with integer labels for each airway region
    label_map : dict
        Mapping from label IDs to anatomical names
    """
    
    print("\n" + "="*60)
    print("AIRWAY VOLUME SEGMENTATION")
    print("="*60)
    
    # Load the airway mask
    nii_img = nib.load(nii_file_path)
    airway_mask = nii_img.get_fdata().astype(bool)
    affine = nii_img.affine
    
    print(f"\nLoaded airway mask: {airway_mask.shape}")
    print(f"Total airway voxels: {np.sum(airway_mask)}")
    
    # Define label map
    label_map = {
        0: 'Background',
        1: 'Trachea',
        2: 'Right Main Bronchus',
        3: 'Left Main Bronchus',
        4: 'RUL Airways',
        5: 'Intermedius',
        6: 'RML Bronchus',
        7: 'RML Airways',
        8: 'RLL Airways',
        9: 'LUL Airways',
        10: 'LLL Airways'
    }
    
    # Initialize labeled volume
    labeled_volume = np.zeros_like(airway_mask, dtype=np.uint8)
    
    # Extract skeleton sets from result
    trachea_set = skeleton_result.get('trachea_set', None)
    right_main_set = skeleton_result.get('right_main_set', None)
    left_main_set = skeleton_result.get('left_main_set', None)
    rul_set = skeleton_result.get('rul_set', None)
    intermedius_main_set = skeleton_result.get('intermedius_main_set', None)
    intermedius_side_branches_set = skeleton_result.get('intermedius_side_branches_set', None)
    rml_skeleton = skeleton_result.get('rml', [])
    rml_set = skeleton_result.get('rml_set', None)
    rll_distal_set = skeleton_result.get('rll_distal_set', None)
    lul_set = skeleton_result.get('lul_set', None)
    lll_set = skeleton_result.get('lll_set', None)
    
    # If sets are None, build them from skeleton lists
    if trachea_set is None:
        print("\n⚠ Skeleton sets not found in result, building from skeleton lists...")
        trachea_skeleton = skeleton_result.get('trachea', [])
        right_main_skeleton = skeleton_result.get('right_main', [])
        left_main_skeleton = skeleton_result.get('left_main', [])
        rul_skeleton = skeleton_result.get('rul', [])
        intermedius_skeleton = skeleton_result.get('intermedius', [])
        rml_skeleton = skeleton_result.get('rml', [])
        lul_skeleton = skeleton_result.get('lul', [])
        lll_skeleton = skeleton_result.get('lll', [])
        
        # Convert lists to sets
        trachea_set = set([tuple(p) for p in trachea_skeleton]) if trachea_skeleton else set()
        right_main_set = set([tuple(p) for p in right_main_skeleton]) if right_main_skeleton else set()
        left_main_set = set([tuple(p) for p in left_main_skeleton]) if left_main_skeleton else set()
        rul_set = set([tuple(p) for p in rul_skeleton]) if rul_skeleton else set()
        intermedius_main_set = set([tuple(p) for p in intermedius_skeleton]) if intermedius_skeleton else set()
        intermedius_side_branches_set = set()  # Will be empty without full skeleton analysis
        rml_set = set([tuple(p) for p in rml_skeleton]) if rml_skeleton else set()
        rll_distal_set = set()  # Will be empty without full skeleton analysis
        lul_set = set([tuple(p) for p in lul_skeleton]) if lul_skeleton else set()
        lll_set = set([tuple(p) for p in lll_skeleton]) if lll_skeleton else set()
        
        print("  ✓ Built skeleton sets from lists")
        print("  ⚠ Note: Distal branch sets (RML/RLL airways) will be incomplete")
        print("  ⚠ Recommendation: Run analyze_right_lung_airways() with visualize=True")
    
    print("\n" + "="*60)
    print("SKELETON SIZES")
    print("="*60)
    print(f"  Trachea: {len(trachea_set)} points")
    print(f"  Right Main: {len(right_main_set)} points")
    print(f"  Left Main: {len(left_main_set)} points")
    print(f"  RUL: {len(rul_set)} points")
    print(f"  Intermedius (main): {len(intermedius_main_set)} points")
    print(f"  Intermedius (side branches → RLL): {len(intermedius_side_branches_set)} points")
    print(f"  RML Bronchus: {len(rml_skeleton)} points")
    print(f"  RML Airways: {len(rml_set)} points")
    print(f"  RLL Airways: {len(rll_distal_set)} points")
    print(f"  LUL: {len(lul_set)} points")
    print(f"  LLL: {len(lll_set)} points")
    
    # Create skeleton masks for distance transform
    print("\n" + "="*60)
    print("CREATING SKELETON MASKS")
    print("="*60)
    
    def create_skeleton_mask(skeleton_set):
        """Create binary mask from skeleton set"""
        mask = np.zeros_like(airway_mask, dtype=bool)
        for point in skeleton_set:
            if len(point) == 3:
                mask[point] = True
        return mask
    
    trachea_mask = create_skeleton_mask(trachea_set)
    right_main_mask = create_skeleton_mask(right_main_set)
    left_main_mask = create_skeleton_mask(left_main_set)
    rul_mask = create_skeleton_mask(rul_set)
    
    # Combine intermedius main + side branches for intermedius region
    intermedius_combined_set = intermedius_main_set | intermedius_side_branches_set
    intermedius_mask = create_skeleton_mask(intermedius_combined_set)
    
    # RML bronchus mask (just the main bronchus path)
    rml_bronchus_mask = create_skeleton_mask(set([tuple(p) for p in rml_skeleton]))
    
    # RML airways mask (distal branches)
    rml_airways_mask = create_skeleton_mask(rml_set)
    
    # RLL airways mask (includes side branches from intermedius)
    rll_airways_mask = create_skeleton_mask(rll_distal_set)
    
    lul_mask = create_skeleton_mask(lul_set)
    lll_mask = create_skeleton_mask(lll_set)
    
    print(f"  Trachea mask: {np.sum(trachea_mask)} voxels")
    print(f"  Right main mask: {np.sum(right_main_mask)} voxels")
    print(f"  Left main mask: {np.sum(left_main_mask)} voxels")
    print(f"  RUL mask: {np.sum(rul_mask)} voxels")
    print(f"  Intermedius mask: {np.sum(intermedius_mask)} voxels")
    print(f"  RML bronchus mask: {np.sum(rml_bronchus_mask)} voxels")
    print(f"  RML airways mask: {np.sum(rml_airways_mask)} voxels")
    print(f"  RLL airways mask: {np.sum(rll_airways_mask)} voxels")
    print(f"  LUL mask: {np.sum(lul_mask)} voxels")
    print(f"  LLL mask: {np.sum(lll_mask)} voxels")
    


    # ============================================
    # HIERARCHICAL ASSIGNMENT (Priority-based)
    # ============================================
    print("\n" + "="*60)
    print("ASSIGNING VOXELS TO AIRWAYS (Hierarchical method)")
    print("="*60)
    
    from scipy.spatial import cKDTree
    
    # Get all airway voxel coordinates
    airway_coords = np.argwhere(airway_mask)
    print(f"  Total airway voxels: {len(airway_coords)}")
    
    # Define hierarchical assignment order with distance thresholds
    # Format: (skeleton_set, label_id, max_distance_mm, label_name)
    assignment_hierarchy = [
        # Stage 1: Main structures (larger distance tolerance)
        (trachea_set, 1, 15, "Trachea"),
        (right_main_set, 2, 10, "Right Main"),
        (left_main_set, 3, 10, "Left Main"),
        
        # Stage 2: Lobar bronchi (medium distance)
        (rul_set, 4, 8, "RUL"),
        (intermedius_combined_set, 5, 8, "Intermedius"),
        (lul_set, 9, 8, "LUL"),
        
        # Stage 3: Specific airways (tighter distance)
        (set([tuple(p) for p in rml_skeleton]), 6, 6, "RML Bronchus"),
        (rml_set, 7, 6, "RML Airways"),
        (rll_distal_set, 8, 6, "RLL Airways"),
        (lll_set, 10, 6, "LLL Airways"),
    ]
    
    # Track which voxels have been assigned
    assigned_mask = np.zeros(len(airway_coords), dtype=bool)
    
    # Get voxel spacing for distance calculations
    voxel_spacing = np.array(nii_img.header.get_zooms())
    mean_spacing = np.mean(voxel_spacing)
    
    # Process each structure in hierarchical order
    for skeleton_set, label_id, max_distance_mm, label_name in assignment_hierarchy:
        if len(skeleton_set) == 0:
            print(f"  {label_name}: No skeleton points, skipping")
            continue
        
        # Build KDTree for this structure
        skeleton_points = np.array(list(skeleton_set))
        tree = cKDTree(skeleton_points)
        
        # Find unassigned voxels
        unassigned_coords = airway_coords[~assigned_mask]
        
        if len(unassigned_coords) == 0:
            print(f"  {label_name}: All voxels already assigned")
            continue
        
        # Query nearest skeleton point for unassigned voxels
        max_distance_voxels = max_distance_mm / mean_spacing
        distances, indices = tree.query(unassigned_coords, k=1, workers=-1)
        
        # Assign voxels within distance threshold
        within_threshold = distances <= max_distance_voxels
        coords_to_assign = unassigned_coords[within_threshold]
        
        # Update labeled volume
        for coord in coords_to_assign:
            labeled_volume[tuple(coord)] = label_id
        
        # Mark as assigned
        unassigned_indices = np.where(~assigned_mask)[0]
        assigned_mask[unassigned_indices[within_threshold]] = True
        
        print(f"  {label_name}: Assigned {len(coords_to_assign)} voxels (max dist: {max_distance_mm}mm)")
    
    # Report unassigned voxels
    remaining_unassigned = np.sum(~assigned_mask)
    print(f"\n  Unassigned voxels remaining: {remaining_unassigned}")
    
    if remaining_unassigned > 0:
        print("  Assigning remaining voxels to nearest structure (no distance limit)...")
        
        # Build combined KDTree with all structures
        all_skeleton_points = []
        all_skeleton_labels = []
        
        for skeleton_set, label_id, _, _ in assignment_hierarchy:
            for point in skeleton_set:
                all_skeleton_points.append(point)
                all_skeleton_labels.append(label_id)
        
        if len(all_skeleton_points) > 0:
            all_skeleton_points = np.array(all_skeleton_points)
            all_skeleton_labels = np.array(all_skeleton_labels)
            tree_all = cKDTree(all_skeleton_points)
            
            unassigned_coords = airway_coords[~assigned_mask]
            distances, indices = tree_all.query(unassigned_coords, k=1, workers=-1)
            
            for i, coord in enumerate(unassigned_coords):
                label = all_skeleton_labels[indices[i]]
                labeled_volume[tuple(coord)] = label
            
            print(f"  ✓ Assigned {len(unassigned_coords)} remaining voxels")
    
    print("  ✓ Hierarchical assignment complete!")

    # ============================================
    # REFINE CARINAL ZONE - Force to Trachea
    # ============================================
    print("\n" + "="*60)
    print("REFINING CARINAL ZONE")
    print("="*60)
    
    # Get carinal zone info from skeleton result
    carinal_zone_center = skeleton_result.get('carinal_zone_center', None)
    carina_zone_radius_cm = skeleton_result.get('carina_zone_radius_cm', 1.0)
    
    if carinal_zone_center is not None:
        zone_radius_voxels = (carina_zone_radius_cm * 10) / np.mean(nii_img.header.get_zooms())
        
        print(f"  Carinal zone center (voxel): {carinal_zone_center}")
        print(f"  Zone radius: {carina_zone_radius_cm} cm ({zone_radius_voxels:.1f} voxels)")
        
        # Find all airway voxels within the carinal zone
        airway_coords = np.argwhere(airway_mask)
        
        # Calculate distance from each voxel to carinal zone center
        distances_to_carina = np.linalg.norm(
            airway_coords - np.array(carinal_zone_center), 
            axis=1
        )
        
        # Find voxels within the zone
        within_zone = distances_to_carina <= zone_radius_voxels
        voxels_in_zone = airway_coords[within_zone]
        
        print(f"  Found {len(voxels_in_zone)} airway voxels within carinal zone")
        
        # Reassign all voxels in zone to trachea (label 1)
        for coord in voxels_in_zone:
            labeled_volume[tuple(coord)] = 1
        
        print("  ✓ Carinal zone voxels reassigned to trachea")
    else:
        print("  ⚠ Carinal zone center not found in skeleton result")
        print("  ⚠ Skipping carinal zone refinement")

    # ============================================
    # POST-PROCESSING: KEEP LARGEST COMPONENTS + ASSIGN ORPHANS
    # ============================================
    print("\n" + "="*60)
    print("POST-PROCESSING: LARGEST COMPONENTS + ORPHAN ASSIGNMENT")
    print("="*60)
    
    def get_largest_connected_component_volume(label_volume, label_id):
        """
        Find and keep only the largest connected component for a given label.
        Returns the cleaned volume with only the largest component.
        """
        # Extract binary mask for this label
        binary_mask = (label_volume == label_id)
        
        if not np.any(binary_mask):
            return label_volume, 0  # ← ADD THIS: return 0 removed voxels
        
        # Label connected components
        labeled_components, num_components = ndimage.label(binary_mask)
        
        if num_components <= 1:
            return label_volume, 0  # ← CHANGE THIS: Already single component, 0 removed
        
        # Find sizes of each component
        component_sizes = ndimage.sum(binary_mask, labeled_components, range(1, num_components + 1))
        
        # Find largest component
        largest_component_id = np.argmax(component_sizes) + 1
        largest_component_size = component_sizes[largest_component_id - 1]
        
        # Keep only largest component
        largest_component_mask = (labeled_components == largest_component_id)
        
        # Remove small components from labeled_volume
        label_volume[binary_mask & ~largest_component_mask] = 0
        
        removed_voxels = int(np.sum(binary_mask) - largest_component_size)
        
        return label_volume, removed_voxels
    
    # 1. Keep only largest component for trachea (label 1)
    print("\n1. Filtering trachea to largest component...")
    labeled_volume, removed = get_largest_connected_component_volume(labeled_volume, 1)
    print(f"   Trachea: removed {removed} voxels from small components")
    
    # 2. Keep only largest component for right main (label 2)
    print("\n2. Filtering right main to largest component...")
    labeled_volume, removed = get_largest_connected_component_volume(labeled_volume, 2)
    print(f"   Right main: removed {removed} voxels from small components")
    
    # 3. Keep only largest component for left main (label 3)
    print("\n3. Filtering left main to largest component...")
    labeled_volume, removed = get_largest_connected_component_volume(labeled_volume, 3)
    print(f"   Left main: removed {removed} voxels from small components")
    
    # ============================================
    # 4. ASSIGN ORPHANED VOXELS TO TOUCHING LABELS
    # ============================================
    print("\n4. Assigning orphaned voxels to touching structures...")
    
    # Find all unlabeled airway voxels (orphans)
    orphaned_voxels = (airway_mask) & (labeled_volume == 0)
    orphan_count = np.sum(orphaned_voxels)
    
    print(f"   Found {orphan_count} orphaned airway voxels")
    
    if orphan_count > 0:
        # Define priority order (most specific to most general)
        label_priority = [7, 8, 10, 4, 9, 6, 5, 2, 3, 1]  # RML Airways → RLL → LLL → RUL → LUL → RML Bronchus → Intermedius → Right Main → Left Main → Trachea
        
        # Get coordinates of orphaned voxels
        orphan_coords = np.argwhere(orphaned_voxels)
        
        orphans_assigned = 0
        
        # For each orphan, find touching labels
        for coord in orphan_coords:
            z, y, x = coord
            
            # Check 26-connected neighbors
            touching_labels = set()
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        
                        nz, ny, nx = z + dz, y + dy, x + dx
                        
                        # Check bounds
                        if (0 <= nz < labeled_volume.shape[0] and
                            0 <= ny < labeled_volume.shape[1] and
                            0 <= nx < labeled_volume.shape[2]):
                            
                            neighbor_label = labeled_volume[nz, ny, nx]
                            if neighbor_label > 0:
                                touching_labels.add(neighbor_label)
            
            # Assign to highest priority touching label
            if touching_labels:
                # Find highest priority label from touching labels
                for priority_label in label_priority:
                    if priority_label in touching_labels:
                        labeled_volume[tuple(coord)] = priority_label
                        orphans_assigned += 1
                        break
            else:
                # No touching labels - assign to trachea as fallback
                labeled_volume[tuple(coord)] = 1
                orphans_assigned += 1
        
        print(f"   ✓ Assigned {orphans_assigned} orphaned voxels to touching structures")
    
    print("\n✓ Post-processing complete!")
    
    # Print results
    print("\n✓ Labeling complete:")
    
    # Print results
    print("\n✓ Labeling complete:")
    for label_id, label_name in label_map.items():
        if label_id > 0:
            voxel_count = np.sum(labeled_volume == label_id)
            print(f"  {label_name} ({label_id}): {voxel_count} voxels")
    
    unlabeled = np.sum((airway_mask) & (labeled_volume == 0))
    print(f"  Unlabeled: {unlabeled} voxels")
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving labeled volume to: {output_path}")
        labeled_nii = nib.Nifti1Image(labeled_volume.astype(np.uint8), affine)
        nib.save(labeled_nii, output_path)
        print("  ✓ Saved successfully")
    
    return labeled_volume, label_map


def visualize_segmented_airways(nii_file_path, labeled_volume, label_map, screenshot_path=None):
    """
    Create 3D visualization of segmented airways using PyVista.
    
    Parameters:
    -----------
    nii_file_path : str
        Path to original airway file
    labeled_volume : ndarray
        Labeled volume from segment_airway_volumes()
    label_map : dict
        Label mapping dictionary
    screenshot_path : str, optional
        Path to save screenshot
    """
    import pyvista as pv
    from skimage import measure
    
    print("\n" + "="*60)
    print("CREATING PYVISTA VISUALIZATION")
    print("="*60)
    
    # Load NIfTI header info
    nii_img = nib.load(nii_file_path)
    voxel_spacing = nii_img.header.get_zooms()
    affine = nii_img.affine
    
    # Initialize plotter
    if screenshot_path:
        plotter = pv.Plotter(off_screen=True)
        plotter.window_size = [1920, 1080]
    else:
        plotter = pv.Plotter()
        plotter.window_size = [1400, 1000]
    
    # Define colors for each region (RGB 0-255)
    color_map = {
        1: [0, 255, 255],      # Cyan - Trachea
        2: [255, 140, 0],      # Dark Orange - Right Main
        3: [255, 0, 0],        # Red - Left Main
        4: [0, 250, 154],      # Medium Spring Green - RUL
        5: [70, 130, 180],     # Steel Blue - Intermedius
        6: [255, 0, 255],      # Magenta - RML Bronchus 
        7: [255, 0, 255],      # Magenta - RML Airways
        8: [0, 128, 0],        # Green - RLL Airways
        9: [135, 206, 250],    # Light Sky Blue - LUL
        10: [0, 0, 128]        # Navy Blue - LLL
    }
    
    print("\nCreating surface meshes for each airway region...")
    
    # Helper function to transform coordinates
    def transform_to_physical(verts_voxel):
        """Transform voxel coordinates to RAS physical space"""
        verts_homogeneous = np.hstack([verts_voxel, np.ones((len(verts_voxel), 1))])
        verts_physical = (affine @ verts_homogeneous.T).T[:, :3]
        return verts_physical
    
    for label_id in range(1, 11):
        label_name = label_map[label_id]
        
        # Create binary mask for this label
        region_mask = (labeled_volume == label_id).astype(np.uint8)
        voxel_count = np.sum(region_mask)
        
        if voxel_count == 0:
            print(f"  {label_name}: No voxels, skipping")
            continue
        
        print(f"  {label_name}: {voxel_count} voxels, creating surface...")
        
        # Use marching cubes to create surface
        try:
            verts, faces, normals, values = measure.marching_cubes(
                region_mask, 
                level=0.5, 
                spacing=voxel_spacing
            )
            
            # Transform vertices to RAS physical space
            verts_physical = transform_to_physical(verts / voxel_spacing)
            
            # Create PyVista mesh
            mesh = pv.PolyData(verts_physical.astype(np.float32))
            
            # Convert faces to PyVista format
            faces_pv = np.hstack([[3] + face.tolist() for face in faces])
            mesh.faces = faces_pv
            
            # Add to plotter with color
            color_rgb = [c/255.0 for c in color_map[label_id]]
            plotter.add_mesh(
                mesh,
                color=color_rgb,
                opacity=0.8,
                label=label_name,
                smooth_shading=True,
                show_edges=False
            )
            
            print(f"    ✓ Mesh created: {len(verts)} vertices, {len(faces)} faces")
            
        except Exception as e:
            print(f"    ⚠ Could not create mesh: {e}")
    
    # Configure plotter appearance
    #plotter.add_legend(size=(0.15, 0.25), loc='upper right')
    plotter.show_grid()
    plotter.set_background('white')
    
    # Add title
    mrn = Path(nii_file_path).stem.replace('.nii', '')
    #plotter.add_text(f'Segmented Airways - MRN {mrn}', position='upper_edge', font_size=14, color='black')
    
    # Set camera position
    # Get center of all meshes
    if len(plotter.renderer.actors) > 0:
        bounds = plotter.renderer.ComputeVisiblePropBounds()
        center = [(bounds[i] + bounds[i+1])/2 for i in range(0, 6, 2)]
        
        # Position camera
        plotter.camera.position = [
            center[0] + 300,
            center[1],
            center[2] - 30
        ]
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 0, 1]
        plotter.reset_camera()
    
    print("\n✓ Visualization ready!")
    
    # Show or save
    if screenshot_path:
        print(f"  Saving screenshot to: {screenshot_path}")
        plotter.screenshot(screenshot_path)
        plotter.close()
        print(f"  ✓ Screenshot saved")
    else:
        print("  Opening interactive window...")
        plotter.show()


def main(nii_file_path, output_dir=None, visualize=True):
    """
    Main pipeline: analyze skeleton and segment airway volumes.
    
    Parameters:
    -----------
    nii_file_path : str
        Path to airway segmentation NIfTI file
    output_dir : str, optional
        Directory to save outputs
    visualize : bool
        Whether to create visualization
    """
    
    # Step 1: Analyze skeleton (from Script 1)
    print("\n" + "="*60)
    print("STEP 1: ANALYZING SKELETON")
    print("="*60)
    
    skeleton_result = analyze_right_lung_airways(
        nii_file_path,
        carina_zone_radius_cm=1,
        lookahead_distance=3,
        visualize=True,  # Must be True to populate skeleton sets
        screenshot_path='temp_skeleton.png'  # Save to temp file to avoid showing window
    )
    
    # Step 2: Segment airway volumes
    print("\n" + "="*60)
    print("STEP 2: SEGMENTING AIRWAY VOLUMES")
    print("="*60)
    
    # Set output path if directory provided
    output_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mrn = Path(nii_file_path).stem.replace('.nii', '')
        output_path = output_dir / f"{mrn}_labeled_airways.nii.gz"
    
    labeled_volume, label_map = segment_airway_volumes(
        nii_file_path,
        skeleton_result,
        output_path=output_path
    )
    
    # Step 3: Visualize
    if visualize:
        print("\n" + "="*60)
        print("STEP 3: CREATING VISUALIZATION")
        print("="*60)
        
        screenshot_path = None
        if output_dir:
            screenshot_path = output_dir / f"{mrn}_segmented_airways.png"
        
        visualize_segmented_airways(
            nii_file_path,
            labeled_volume,
            label_map,
            screenshot_path=screenshot_path
        )
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    
    return labeled_volume, label_map, skeleton_result


if __name__ == "__main__":
    # Example usage
    nii_file_path = r"path/to/airway_segmentation.nii.gz"
    output_dir = r"path/to/segmented_output"
    
    # Run the full pipeline
    labeled_volume, label_map, skeleton_result = main(
        nii_file_path,
        output_dir=None,
        visualize=True
    )
    
    print("\nDone!")