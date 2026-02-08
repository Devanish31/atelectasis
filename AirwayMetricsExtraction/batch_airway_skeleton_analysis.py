import os
import numpy as np
import pandas as pd
from pathlib import Path
from analyze_airway_skeleton import analyze_right_lung_airways
from multiprocessing import Pool, Lock
from functools import partial

# Global lock for thread-safe Excel writing
excel_lock = Lock()


def process_single_file(nii_file, output_folder, excel_path):
    """
    Process a single NIfTI file - designed for parallel execution
    """
    mrn = nii_file.stem.replace('.nii', '')
    screenshot_path = os.path.join(output_folder, f"{mrn}_airways.png")
    
    print(f"Processing {mrn}...")
    
    result_dict = {
        'MRN': mrn,
        'Inference': 'Not started'
    }
    
    try:
        # Run analysis with screenshot
        result = analyze_right_lung_airways(
            str(nii_file),
            carina_zone_radius_cm=1,
            lookahead_distance=3,
            visualize=True,
            screenshot_path=screenshot_path,
            title=f"CT Airway Analysis : MRN {mrn}"
        )
        
        if result is not None:
            print(f"  ✓ {mrn}: Saved screenshot")
            
            # Calculate point counts from sets
            trachea_points = len(result.get('trachea_set', [])) if result.get('trachea_set') else 0
            right_main_points = len(result.get('right_main_set', [])) if result.get('right_main_set') else 0
            rul_points = len(result.get('rul_set', [])) if result.get('rul_set') else 0
            intermedius_points = len(result.get('intermedius_main_set', [])) if result.get('intermedius_main_set') else 0
            rml_points = len(result.get('rml', [])) if result.get('rml') else 0
        
            
            # Determine inference
            if trachea_points > 0 and right_main_points == 0:
                inference = "Carina not detected"
            elif right_main_points > 0 and rul_points == 0:
                inference = "RUL not detected"
            elif rul_points > 0 and intermedius_points == 0:
                inference = "Intermedius not detected"
            elif intermedius_points > 0 and rml_points == 0:
                inference = "RML not detected"
            else:
                inference = "Complete"
            
            result_dict.update({
                # Airway point counts
                'RML_Points': rml_points,
                
                # RML Angles
                'RML_Full_Angle_deg': result.get('rml_full_angle'),
                
                # RML Plane Angles
                'RML_Full_Sagittal_Angle_deg': result.get('rml_full_sagittal_angle'),
                'RML_Full_Coronal_Angle_deg': result.get('rml_full_coronal_angle'),
                'RML_Full_Transverse_Angle_deg': result.get('rml_full_transverse_angle'),
                
                # RML Cross-sections
                'RML_CrossSection_Orifice_mm2': result.get('rml_cross_section_orifice'),
                'RML_CrossSection_Mean_mm2': result.get('rml_cross_section_mean'),
                'RML_CrossSection_Median_mm2': result.get('rml_cross_section_median'),
                'RML_CrossSection_Min_mm2': result.get('rml_cross_section_min'),

                # RML "Real" Measurements (between filtered cross-sections)
                'RML_Real_Curved_Length_mm_5': result.get('rml_real_curved_length_mm_5'),
                'RML_Volume_mm3': result.get('rml_volume_mm3_5'),
                
                # Metadata
                'Inference': inference,
            })
            
            print(f"  ✓ {mrn}: {inference}")
            
        else:
            print(f"  ✗ {mrn}: Analysis returned None")
            result_dict['Inference'] = 'Analysis failed'
            
    except Exception as e:
        print(f"  ✗ {mrn}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        result_dict['Inference'] = f'Error: {str(e)[:50]}'
    
    return result_dict


def update_excel_safe(results_list, excel_path):
    """
    Thread-safe Excel update
    """
    with excel_lock:
        df = pd.DataFrame(results_list)
        df.to_excel(excel_path, index=False)


def batch_analyze_and_screenshot():
    """
    Batch process all NIfTI files with parallel processing
    """
    
    # Paths
    input_folder = r"path/to/airway_predictions"
    output_folder = r"path/to/airway_analysis_output"
    excel_path = os.path.join(output_folder, "rml_analysis_summary.xlsx")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all .nii.gz files
    nii_files = list(Path(input_folder).glob("*.nii.gz"))
    
    print("="*80)
    print(f"RML DETECTION BATCH ANALYSIS")
    print(f"Found {len(nii_files)} NIfTI files to process")
    print(f"Using 4 parallel processes")
    print("="*80)
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_file,
        output_folder=output_folder,
        excel_path=excel_path,
    )
    
    # Process in parallel with 4 workers
    results_list = []
    
    with Pool(processes=4) as pool:
        # Use imap to get results as they complete
        for i, result_dict in enumerate(pool.imap(process_func, nii_files), 1):
            results_list.append(result_dict)
            
            # Update Excel after each completion
            update_excel_safe(results_list, excel_path)
            print(f"Progress: {i}/{len(nii_files)} cases completed\n")
    
    # Final summary
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total files: {len(nii_files)}")
    print(f"Completed: {len(results_list)}")
    
    print(f"\nScreenshots saved to: {output_folder}")
    print(f"Excel summary: {excel_path}")
    
    # Print inference summary
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    inference_counts = df['Inference'].value_counts()
    for inference, count in inference_counts.items():
        print(f"  {inference}: {count}")
    
    # RML-specific statistics
    print("\n" + "="*80)
    print("RML DETECTION STATISTICS")
    print("="*80)
    
    # Cases that reached intermedius stage
    reached_intermedius = df[df['Intermedius_Points'] > 0]
    print(f"Cases with intermedius detected: {len(reached_intermedius)}/{len(df)}")
    
    if len(reached_intermedius) > 0:
        # RML detection rate
        rml_detected = df[df['RML_Points'] > 0]
        print(f"RML detected: {len(rml_detected)}/{len(reached_intermedius)} ({100*len(rml_detected)/len(reached_intermedius):.1f}%)")
        print(f"RML NOT detected: {len(reached_intermedius) - len(rml_detected)}/{len(reached_intermedius)} ({100*(len(reached_intermedius) - len(rml_detected))/len(reached_intermedius):.1f}%)")
        
        # RML length statistics
        if len(rml_detected) > 0:
            valid_lengths = rml_detected[rml_detected['RML_Length_mm'].notna()]
            if len(valid_lengths) > 0:
                print(f"\nRML LENGTH STATISTICS (n={len(valid_lengths)}):")
                print(f"  Mean: {valid_lengths['RML_Length_mm'].mean():.1f} mm")
                print(f"  Median: {valid_lengths['RML_Length_mm'].median():.1f} mm")
                print(f"  Range: {valid_lengths['RML_Length_mm'].min():.1f} - {valid_lengths['RML_Length_mm'].max():.1f} mm")
                print(f"  Std Dev: {valid_lengths['RML_Length_mm'].std():.1f} mm")
    
    # Airway length statistics
    print("\n" + "="*80)
    print("AIRWAY LENGTH STATISTICS")
    print("="*80)
    complete_cases = df[df['Inference'] == 'Complete']
    if len(complete_cases) > 0:
        print(f"Fully complete cases: {len(complete_cases)}/{len(df)} ({100*len(complete_cases)/len(df):.1f}%)")
        
        for airway in ['Trachea', 'Right_Main', 'Left_Main', 'Intermedius']:
            length_col = f'{airway}_Length_mm'
            points_col = f'{airway}_Points'
            
            valid_lengths = complete_cases[complete_cases[length_col].notna()]
            if len(valid_lengths) > 0:
                print(f"\n{airway.replace('_', ' ')}:")
                print(f"  Mean length: {valid_lengths[length_col].mean():.1f} mm ({valid_lengths[points_col].mean():.0f} points)")
                print(f"  Median length: {valid_lengths[length_col].median():.1f} mm")
                print(f"  Range: {valid_lengths[length_col].min():.1f} - {valid_lengths[length_col].max():.1f} mm")


if __name__ == "__main__":
    batch_analyze_and_screenshot()