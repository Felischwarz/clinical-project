import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# we now want to check if the registration is correct

first_ct_data = r"D:\tum_clinical_project\isles24_combined\derivatives\sub-stroke0001\ses-01"
first_lesion_data = r"D:\tum_clinical_project\isles24_combined\derivatives\sub-stroke0001\ses-02"

perfusion_data = r"D:\tum_clinical_project\isles24_combined\derivatives\sub-stroke0001\ses-01\perfusion-maps"

# load images
ct_images = [os.path.join(first_ct_data, f) for f in os.listdir(first_ct_data) if f.endswith('.nii.gz')]
# load perfusion images
perfusion_images = [os.path.join(perfusion_data, f) for f in os.listdir(perfusion_data) if f.endswith('.nii.gz')]


lesion_images = [os.path.join(first_lesion_data, f) for f in os.listdir(first_lesion_data) if f.endswith('.nii.gz')]

#check the affine matrix of the images
for ct_image_path in ct_images:
    print(f"Filename: {os.path.basename(ct_image_path)}")
    ct_image = sitk.ReadImage(ct_image_path)
    print(ct_image.GetOrigin())
    print(ct_image.GetSpacing())
    print(ct_image.GetDirection())
    print(ct_image.GetSize())

for lesion_image_path in lesion_images:
    print(f"Filename: {os.path.basename(lesion_image_path)}")
    lesion_image = sitk.ReadImage(lesion_image_path)
    print(lesion_image.GetOrigin())
    print(lesion_image.GetSpacing())
    print(lesion_image.GetDirection())
    print(lesion_image.GetSize())


for perfusion_image_path in perfusion_images:
    print(f"Filename: {os.path.basename(perfusion_image_path)}")
    perfusion_image = sitk.ReadImage(perfusion_image_path)
    print(perfusion_image.GetOrigin())
    print(perfusion_image.GetSpacing())
    print(perfusion_image.GetDirection())
    print(perfusion_image.GetSize())

