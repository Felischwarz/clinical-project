"""
Script to check the dimensions of all NIfTI files in the dataset.
This will help identify which files have different dimensions that could cause collation issues.
"""

import os
import glob
import nibabel as nib
from collections import defaultdict

# Use the same data directory as in main.py
data_dir = r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES"

# Dictionaries to store dimensions by file type
dimensions = defaultdict(list)
file_paths = defaultdict(list)

print("Checking all NIfTI files in the dataset...\n")

# Recursively scan all .nii.gz files
for root, _, _ in os.walk(data_dir):
    for nii_file in glob.glob(os.path.join(root, "*.nii.gz")):
        try:
            # Load the NIfTI file
            img = nib.load(nii_file)
            data_shape = img.get_fdata().shape
            
            # Extract file type from filename
            basename = os.path.basename(nii_file)
            if "cta" in basename:
                file_type = "CTA"
            elif "ctp" in basename:
                file_type = "CTP"
            elif "cbf" in basename:
                file_type = "CBF"
            elif "cbv" in basename:
                file_type = "CBV"
            elif "mtt" in basename:
                file_type = "MTT"
            elif "tmax" in basename:
                file_type = "TMAX"
            elif "lesion-msk" in basename:
                file_type = "LESION_MASK"
            else:
                file_type = "OTHER"
            
            # Store dimensions and file path
            dimensions[file_type].append(data_shape)
            file_paths[file_type].append(nii_file)
            
        except Exception as e:
            print(f"Error loading {nii_file}: {e}")

# Print summary of dimensions by file type
print("Summary of file dimensions by type:\n")
for file_type, shapes in dimensions.items():
    print(f"{file_type} files:")
    # Group by unique shapes
    unique_shapes = {}
    for i, shape in enumerate(shapes):
        if shape not in unique_shapes:
            unique_shapes[shape] = []
        unique_shapes[shape].append(file_paths[file_type][i])
    
    # Print each unique shape and count
    for shape, paths in unique_shapes.items():
        print(f"  - Shape {shape}: {len(paths)} files")
        # Print a sample of files with this shape (max 2)
        for path in paths[:2]:
            print(f"    - Example: {path}")
    print()

# Print potential problem files
print("\nPotential problem files (comparing across types):")
if "CTA" in dimensions and "LESION_MASK" in dimensions:
    cta_shapes = set(dimensions["CTA"])
    mask_shapes = set(dimensions["LESION_MASK"])
    
    print(f"\nCTA unique shapes: {cta_shapes}")
    print(f"LESION_MASK unique shapes: {mask_shapes}")
    
    if cta_shapes != mask_shapes:
        print("\nWarning: CTA and LESION_MASK files have different shapes!")
        
        # Print examples of mismatched pairs
        print("\nExamples of potentially mismatched pairs:")
        for i, cta_file in enumerate(file_paths["CTA"][:5]):  # Limit to first 5
            cta_shape = dimensions["CTA"][i]
            subject = os.path.basename(os.path.dirname(os.path.dirname(cta_file)))
            
            # Find matching mask for this subject
            matching_masks = [p for p in file_paths["LESION_MASK"] 
                              if subject.lower() in p.lower()]
            
            if matching_masks:
                mask_file = matching_masks[0]
                mask_idx = file_paths["LESION_MASK"].index(mask_file)
                mask_shape = dimensions["LESION_MASK"][mask_idx]
                
                print(f"Subject: {subject}")
                print(f"  CTA: {cta_file} - Shape: {cta_shape}")
                print(f"  Mask: {mask_file} - Shape: {mask_shape}")
                if cta_shape != mask_shape:
                    print(f"  MISMATCH DETECTED!")
                print() 