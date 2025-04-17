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
)

from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

# For testing purposes, limit the number of subjects
MAX_SUBJECTS = 3  # Set to None to use all subjects

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
    SpatialPadd(keys=["image", "label"], spatial_size=[256, 256, 80]),
    RandRotated(keys=["image", "label"], range_x=15, range_y=15, range_z=15, prob=0.3),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    ToTensord(keys=["image", "label"]),
    # make sure the spacing and orientation are the same for the image and label
    Spacingd(keys=["image", "label"], pixdim=[1.0, 1.0, 1.0]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandSpatialCropd(keys=["image", "label"], roi_size=[600, 600, 80], random_size=False),
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
loss_function = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

#initialize optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

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
        metric.update(outputs, labels)
        
        batch_count += 1
        print(f"  Batch {batch_count} completed in {time.time() - batch_start:.2f}s")
    
    epoch_loss /= len(train_loader)
    epoch_dice = metric.compute()       
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, Time: {time.time() - epoch_start:.2f}s")


#save model
print(f"Saving model at {time.strftime('%H:%M:%S')}")
torch.save(unet.state_dict(), "unet_model.pth")
print("Training completed!")


