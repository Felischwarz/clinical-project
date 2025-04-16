import sys 
import nibabel as nib 
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def check_normalization(data, tol_mean=0.1, tol_std=0.1): 
    """ Checks if the data is normalized. 
    - If all values are between 0 and 1, it is assumed that min-max normalization has been applied.
    - If the mean is close to 0 (± tol_mean) and the standard deviation is close to 1 (± tol_std), 
      it is assumed that the data is z-score normalized.
    Otherwise, it is determined that no normalization is present. """
    if np.all(data >= 0) and np.all(data <= 1): 
        return "Min-Max Normalization"
    elif np.abs(np.mean(data)) < tol_mean and np.abs(np.std(data) - 1) < tol_std: 
        return "Z-Score Normalization"
    else: 
        return "No Normalization"
    
def normalize_data(data, method="min-max"): 
    """ Normalizes the data based on the specified method. """
    if method == "min-max": 
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == "z-score":    
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:  # Prevent division by zero
            return np.zeros_like(data, dtype=np.float32)
        return (data - mean_val) / std_val
    else: 
        raise ValueError(f"Invalid normalization method: {method}") 

def visualize_normalization(data, normalized_data, slice_idx=None, method="z-score", output_file=None):
    """
    Visualizes the original data and normalized data side by side.
    
    Args:
        data: Original data (3D array)
        normalized_data: Normalized data (3D array)
        slice_idx: Index of the slice to be visualized. If None, the middle is chosen.
        method: Normalization method used (for the title)
        output_file: Path to save the visualization. If None, the visualization is displayed.
    """
    if slice_idx is None:
        slice_idx = data.shape[2] // 2  # Middle slice in z-direction
    
    # Ensure the slice is within valid range
    if slice_idx < 0 or slice_idx >= data.shape[2]:
        slice_idx = data.shape[2] // 2
    
    # Extract the slice
    original_slice = data[:, :, slice_idx]
    normalized_slice = normalized_data[:, :, slice_idx]
    
    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig)
    
    # Image plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])  # Difference image
    
    # Histogram plots
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])  # Comparison of the two histograms
    
    # Original image
    im1 = ax1.imshow(original_slice, cmap='gray')
    ax1.set_title(f'Original (Slice {slice_idx})')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Normalized image
    im2 = ax2.imshow(normalized_slice, cmap='gray')
    ax2.set_title(f'{method} Normalized (Slice {slice_idx})')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Difference image
    im3 = ax3.imshow(original_slice - normalized_slice, cmap='coolwarm')
    ax3.set_title('Difference')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Histograms
    ax4.hist(original_slice.flatten(), bins=50, alpha=0.7)
    ax4.set_title('Histogram Original')
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Frequency')
    
    ax5.hist(normalized_slice.flatten(), bins=50, alpha=0.7)
    ax5.set_title(f'Histogram {method}')
    ax5.set_xlabel('Intensity')
    
    # Overlaid histogram for comparison
    ax6.hist(original_slice.flatten(), bins=50, alpha=0.5, label='Original')
    ax6.hist(normalized_slice.flatten(), bins=50, alpha=0.5, label='Normalized')
    ax6.set_title('Histogram Comparison')
    ax6.set_xlabel('Intensity')
    ax6.legend()
    
    # Display statistics
    original_stats = f"Original - Min: {np.min(original_slice):.2f}, Max: {np.max(original_slice):.2f}, Mean: {np.mean(original_slice):.2f}, Std: {np.std(original_slice):.2f}"
    normalized_stats = f"Normalized - Min: {np.min(normalized_slice):.2f}, Max: {np.max(normalized_slice):.2f}, Mean: {np.mean(normalized_slice):.2f}, Std: {np.std(normalized_slice):.2f}"
    
    plt.figtext(0.5, 0.01, original_stats + "\n" + normalized_stats, ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Comparison: Original vs {method} Normalization', fontsize=16)
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def normalize_nifti_file(input_file, output_file, method="z-score", visualize=False, visualization_output=None): 
    """ Normalizes a NIfTI image based on the specified method and saves the result to a new file. """
    # Read the NIfTI image with float32 instead of float64 to save memory
    img = nib.load(input_file)
    data = img.get_fdata(dtype=np.float32)  
    
    # Check if the file is already normalized
    normalization_status = check_normalization(data)
    if normalization_status != "No Normalization":
        print(f"File {input_file} is already {normalization_status} normalized.")
        return
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Normalize the data
    normalized_data = normalize_data(data, method)
    
    # Visualize the normalization if requested
    if visualize:
        vis_output = visualization_output
        if vis_output is None and output_file.endswith('.nii.gz'):
            vis_output = output_file.replace('.nii.gz', f'_{method}_visualization.png')
        
        visualize_normalization(data, normalized_data, method=method, output_file=vis_output)
        if vis_output:
            print(f"Visualization saved to {vis_output}")
    
    # Create a new NIfTI image with the normalized data
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)   
    
    # Save the normalized image
    nib.save(normalized_img, output_file)
    
    print(f"Normalized image saved to {output_file}.")
    
if __name__ == "__main__": 
    '''
    Folder structure: 
    D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES\SUB-STROKE0001
├───ses-01
│   │   sub-stroke0001_ses-01_space-ncct_cta.nii.gz
│   │   sub-stroke0001_ses-01_space-ncct_ctp.nii.gz
│   │
│   └───perfusion-maps
│           sub-stroke0001_ses-01_space-ncct_cbf.nii.gz
│           sub-stroke0001_ses-01_space-ncct_cbv.nii.gz
│           sub-stroke0001_ses-01_space-ncct_mtt.nii.gz
│           sub-stroke0001_ses-01_space-ncct_tmax.nii.gz
│
└───ses-02
        sub-stroke0001_ses-02_lesion-msk.nii.gz

    '''     
    input_dir = r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES"
    output_dir = r"D:\TUM_CLINICAL_PROJECT\output\normalized_data"
    visualize_dir = r"D:\TUM_CLINICAL_PROJECT\output\visualizations"
    
    # Ensure the output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)
    
    # Process all sessions, check the files and normalize them
    for subject in os.listdir(input_dir): 
        subject_path = os.path.join(input_dir, subject)
        if os.path.isdir(subject_path): 
            # Create subject directory in the output folder
            subject_output_path = os.path.join(output_dir, subject)
            os.makedirs(subject_output_path, exist_ok=True)
            
            # Create visualization directory for this subject
            subject_vis_path = os.path.join(visualize_dir, subject)
            os.makedirs(subject_vis_path, exist_ok=True)
            
            for session in os.listdir(subject_path): 
                session_path = os.path.join(subject_path, session)
                if os.path.isdir(session_path): 
                    # Create session directory in the output folder
                    session_output_path = os.path.join(subject_output_path, session)
                    os.makedirs(session_output_path, exist_ok=True)
                    
                    # Create session visualization directory
                    session_vis_path = os.path.join(subject_vis_path, session)
                    os.makedirs(session_vis_path, exist_ok=True)
                    
                    # Process files in the main directory of the session
                    for file in os.listdir(session_path): 
                        file_path = os.path.join(session_path, file)
                        if os.path.isfile(file_path) and file.endswith(".nii.gz"): 
                            output_file_path = os.path.join(session_output_path, file)
                            vis_output_path = os.path.join(session_vis_path, file.replace('.nii.gz', '_visualization.png'))
                            
                            # Check if it's a lesion mask
                            if "lesion" in file.lower() or "msk" in file.lower():
                                # Don't normalize lesion masks, just copy them
                                print(f"Lesion mask {file_path} will not be normalized.")
                                # Copy the file if it doesn't exist yet
                                if not os.path.exists(output_file_path):
                                    import shutil
                                    shutil.copy2(file_path, output_file_path)
                                    print(f"Lesion mask copied to {output_file_path}.")
                            else:
                                try:
                                    normalize_nifti_file(file_path, output_file_path, "z-score", 
                                                        visualize=True, visualization_output=vis_output_path)
                                except Exception as e:
                                    print(f"Error normalizing {file_path}: {e}")
                        
                    # Check if a perfusion-maps directory exists and process its files
                    perfusion_maps_path = os.path.join(session_path, "perfusion-maps")
                    if os.path.isdir(perfusion_maps_path):
                        perfusion_maps_output_path = os.path.join(session_output_path, "perfusion-maps")
                        os.makedirs(perfusion_maps_output_path, exist_ok=True)
                        
                        perfusion_maps_vis_path = os.path.join(session_vis_path, "perfusion-maps")
                        os.makedirs(perfusion_maps_vis_path, exist_ok=True)
                        
                        for file in os.listdir(perfusion_maps_path):
                            file_path = os.path.join(perfusion_maps_path, file)
                            if os.path.isfile(file_path) and file.endswith(".nii.gz"):
                                output_file_path = os.path.join(perfusion_maps_output_path, file)
                                vis_output_path = os.path.join(perfusion_maps_vis_path, file.replace('.nii.gz', '_visualization.png'))
                                
                                try:
                                    normalize_nifti_file(file_path, output_file_path, "z-score", 
                                                        visualize=True, visualization_output=vis_output_path)
                                except Exception as e:
                                    print(f"Error normalizing {file_path}: {e}")
                    
    print("Normalization and visualization completed.")


