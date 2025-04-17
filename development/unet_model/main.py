'''
This script is used to train a U-Net model and the MONAI pipeline for the task of predicting MRI-based lesion masks from CT data.
Dice loss and Dice metric are used as the loss function and metric respectively.
'''

data_dir = r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES"

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


import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged, 
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated, 
    RandFlipd,
    ToTensord,
    SpatialPadd,  # Add spatial padding to ensure consistent sizes
    Compose, 

    Spacingd,
    Orientationd,

    CropForegroundd,
    RandSpatialCropd,
    DivisiblePadd,
)

from monai.data import DataLoader, Dataset, pad_list_data_collate
from torch.nn import BCEWithLogitsLoss
from monai.metrics import DiceMetric

# For testing purposes, limit the number of subjects
MAX_SUBJECTS = None  # Set to None to use all subjects

print(f"Starting data loading at {time.strftime('%H:%M:%S')}")

# Create pairs of input images and target masks
data_pairs = []
subject_count = 0
for subject in os.listdir(data_dir): 
    subject_path = os.path.join(data_dir, subject)
    if os.path.isdir(subject_path):
        # Look for input images in ses-01
        input_files = {}
        
        # Find the input image (CT) in session 01
        session1_path = os.path.join(subject_path, "ses-01")
        if os.path.isdir(session1_path):
            for file in os.listdir(session1_path):
                if file.endswith("_cta.nii.gz"):  # Using CTA as input
                    input_files["image"] = os.path.join(session1_path, file)
        
        # Find the mask in session 02
        session2_path = os.path.join(subject_path, "ses-02")
        if os.path.isdir(session2_path):
            for file in os.listdir(session2_path):
                if "lesion-msk" in file:
                    input_files["label"] = os.path.join(session2_path, file)
        
        # If we have both input and mask, add to our dataset
        if "image" in input_files and "label" in input_files:
            data_pairs.append(input_files)
            subject_count += 1
            if MAX_SUBJECTS is not None and subject_count >= MAX_SUBJECTS:
                break

print(f"Found {len(data_pairs)} valid image-mask pairs")
print(f"Data collection completed at {time.strftime('%H:%M:%S')}")

# Define dictionary transforms for paired data
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image", "label"]),
    # Add spatial padding to enforce consistent dimensions
    SpatialPadd(keys=["image", "label"], spatial_size=[256, 256, 64]),
    RandRotated(keys=["image", "label"], range_x=15, range_y=15, range_z=15, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    ToTensord(keys=["image", "label"]),
    # make sure the spacing and orientation are the same for the image and label
    Spacingd(keys=["image", "label"], pixdim=[1.0, 1.0, 1.0]), #use the same spacing for every voxel

    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandSpatialCropd(keys=["image", "label"], roi_size=[256, 256, 64], random_size=False),

    DivisiblePadd(keys=["image", "label"], k=16), #voxel shape (in Anzahl Voxel) must be divisible by k
    Orientationd(keys=["image", "label"], axcodes="RAS"), #use the same orientation for every voxel
])


print(f"Creating dataset at {time.strftime('%H:%M:%S')}")
train_dataset = Dataset(data=data_pairs, transform=train_transforms)

# Reduce batch size to 1 to avoid memory issues
train_loader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=0,  # No multiprocessing to simplify debugging
)
print(f"DataLoader created at {time.strftime('%H:%M:%S')}")


#initialize unet model from monai
unet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)

# run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
unet.to(device)

#initialize loss function and metric
loss_function = BCEWithLogitsLoss()
metric = DiceMetric(include_background=True, reduction="mean")

#initialize optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

# Function to visualize predictions and ground truth
def visualize_prediction(input_image, predicted_mask, ground_truth, slice_idx=None):
    # Convert tensors to numpy arrays
    input_np = input_image.detach().cpu().numpy()[0, 0]  # Remove batch and channel dims
    pred_np = predicted_mask.detach().cpu().numpy()[0, 0]
    gt_np = ground_truth.detach().cpu().numpy()[0, 0]
    
    # Print shape information for debugging
    print(f"3D Volume shape: {input_np.shape}")
    
    # If slice_idx is not provided, use the middle slice
    if slice_idx is None:
        slice_idx = input_np.shape[2] // 2
    
    # Get the slices
    input_slice = input_np[:, :, slice_idx]
    pred_slice = pred_np[:, :, slice_idx]
    gt_slice = gt_np[:, :, slice_idx]
    
    # Print slice shape information
    print(f"2D Slice shape: {input_slice.shape}")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the input image with aspect='equal' to maintain proportions
    axes[0].imshow(input_slice, cmap='gray', aspect='equal')
    axes[0].set_title(f'Input Image (Shape: {input_slice.shape})')
    axes[0].axis('off')
    
    # Plot the predicted mask with aspect='equal'
    axes[1].imshow(pred_slice, cmap='hot', aspect='equal')
    axes[1].set_title(f'Predicted Mask (Shape: {pred_slice.shape})')
    axes[1].axis('off')
    
    # Plot the ground truth mask with aspect='equal'
    axes[2].imshow(gt_slice, cmap='hot', aspect='equal')
    axes[2].set_title(f'Ground Truth Mask (Shape: {gt_slice.shape})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also show alternative views (axial, coronal, sagittal) for better understanding
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Choose middle slices for each plane
    axial_idx = input_np.shape[2] // 2
    coronal_idx = input_np.shape[1] // 2
    sagittal_idx = input_np.shape[0] // 2
    
    # Axial view (already shown above, but included for completeness)
    axes[0, 0].imshow(input_np[:, :, axial_idx], cmap='gray', aspect='equal')
    axes[0, 0].set_title('Input - Axial (Top-Down)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_np[:, :, axial_idx], cmap='hot', aspect='equal')
    axes[0, 1].set_title('Prediction - Axial')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_np[:, :, axial_idx], cmap='hot', aspect='equal')
    axes[0, 2].set_title('Ground Truth - Axial')
    axes[0, 2].axis('off')
    
    # Coronal view
    axes[1, 0].imshow(input_np[:, coronal_idx, :].T, cmap='gray', aspect='equal')
    axes[1, 0].set_title('Input - Coronal (Front-Back)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_np[:, coronal_idx, :].T, cmap='hot', aspect='equal')
    axes[1, 1].set_title('Prediction - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(gt_np[:, coronal_idx, :].T, cmap='hot', aspect='equal')
    axes[1, 2].set_title('Ground Truth - Coronal')
    axes[1, 2].axis('off')
    
    # Sagittal view
    axes[2, 0].imshow(input_np[sagittal_idx, :, :].T, cmap='gray', aspect='equal')
    axes[2, 0].set_title('Input - Sagittal (Side)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(pred_np[sagittal_idx, :, :].T, cmap='hot', aspect='equal')
    axes[2, 1].set_title('Prediction - Sagittal')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(gt_np[sagittal_idx, :, :].T, cmap='hot', aspect='equal')
    axes[2, 2].set_title('Ground Truth - Sagittal')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# If you want to run a quick test first to ensure everything works
TEST_RUN = True
if TEST_RUN:
    print(f"Testing data loading at {time.strftime('%H:%M:%S')}")
    try:
        test_batch = next(iter(train_loader))
        print(f"Test batch loaded successfully")
        print(f"Image shape: {test_batch['image'].shape}")
        print(f"Label shape: {test_batch['label'].shape}")
        print(f"Test completed at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)  # Exit if test fails

#training loop
print(f"Starting training at {time.strftime('%H:%M:%S')}")
num_epochs = 5  # Reduced for testing
for epoch in range(num_epochs):
    unet.train()
    epoch_loss = 0
    metric.reset()
    batch_count = 0
    
    epoch_start = time.time()
    for batch in train_loader:
        batch_start = time.time()
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()    

        epoch_loss += loss.item()
        pred = torch.sigmoid(outputs) > 0.5
        pred = pred.float()
        metric(pred, labels)  # Use __call__ method instead of update
        
        batch_count += 1
        print(f"  Batch {batch_count} completed in {time.time() - batch_start:.2f}s")
    
    epoch_loss /= len(train_loader)
    # Get metric result (mean Dice score)
    epoch_dice = metric.aggregate().item()
    # Reset metric for next epoch
    metric.reset()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, Time: {time.time() - epoch_start:.2f}s")

#save model
print(f"Saving model at {time.strftime('%H:%M:%S')}")
torch.save(unet.state_dict(), "unet_model.pth")
print("Training completed!")

# Add a testing section to visualize predictions on test data
print("Running model on test data for visualization...")
unet.eval()
with torch.no_grad():
    for i, test_batch in enumerate(train_loader):
        if i >= 5:  # Show 5 test samples
            break
            
        test_inputs = test_batch["image"].to(device)
        test_labels = test_batch["label"].to(device)
        test_outputs = unet(test_inputs)
        test_pred = torch.sigmoid(test_outputs) > 0.5
        test_pred = test_pred.float()
        
        print(f"Test sample {i+1}:")
        # Visualize middle slice
        visualize_prediction(test_inputs, test_pred, test_labels)
        
        # Find slices with lesions and visualize one of them
        gt_np = test_labels.detach().cpu().numpy()[0, 0]
        lesion_slices = np.where(np.sum(gt_np, axis=(0, 1)) > 0)[0]
        if len(lesion_slices) > 0:
            print(f"Test sample {i+1} slice with lesion:")
            visualize_prediction(test_inputs, test_pred, test_labels, 
                               slice_idx=lesion_slices[len(lesion_slices)//2])


