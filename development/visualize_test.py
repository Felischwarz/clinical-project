import nibabel as nib # read nifti files
import os
import matplotlib.pyplot as plt
import numpy as np

path = r"D:\tum_clinical_project\isles24_combined\derivatives\sub-stroke0001\ses-01"

# Get all .gz files in the directory
gz_files = [f for f in os.listdir(path) if f.endswith('.gz')]

print(len(gz_files))
# Create a list to store loaded files
perfusion_maps = []

# Load each .gz file directly
for gz_file in gz_files:
    gz_path = os.path.join(path, gz_file)
    img = nib.load(gz_path)
    perfusion_maps.append(img)

# visualize the perfusion maps
for i, perfusion_map in enumerate(perfusion_maps):
    print(f"Visualizing perfusion map {i+1}")
    # print file name
    print(gz_file)
    plt.figure(figsize=(10, 10))
    # Get the data and select only one slice for 2D visualization
    data = perfusion_map.get_fdata()
    slice_data = data[:, :, 10]  # Select slice 10
    plt.imshow(slice_data, cmap='gray')
    plt.colorbar()
    plt.title(f"Perfusion Map {i+1}")
    plt.show()
    # Close the figure to free memory
    plt.close()

print(len(perfusion_maps))