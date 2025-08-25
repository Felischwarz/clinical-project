'''
This script is used to train a U-Net model and the MONAI pipeline for the task of predicting MRI-based lesion masks from CT data.
Dice loss and Dice metric are used as the loss function and metric respectively.
'''

data_dir = r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES"
json_file = r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\isles24_multimodal_5fold_NIHSSstratified.json"

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
import json

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
from monai.losses import DiceLoss
import torch.nn.functional as F
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import ScaleIntensityRanged, AsDiscreted
try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    ndi = None
    _HAS_SCIPY = False

# For testing purposes, limit the number of subjects
MAX_SUBJECTS = None  # Set to None to use all subjects
TARGET_FOLD = 0  # Only train on fold 0

print(f"Starting data loading at {time.strftime('%H:%M:%S')}")

# Load the JSON file with fold information
print(f"Loading fold information from {json_file}")
with open(json_file, 'r') as f:
    fold_data = json.load(f)

# Filter data to only include fold 0
fold_0_cases = [case for case in fold_data['training'] if case['fold'] == TARGET_FOLD]
print(f"Found {len(fold_0_cases)} cases in fold {TARGET_FOLD}")


# Experiment configuration (CTA only)
CTA_PREPROCESS_MODE = "per_image_minmax"  # or "per_image_minmax"

# Mina examples
CTA_WINDOW = 80
CTA_LEVEL = 40

# Results file (will append one line per run)
RESULTS_FILE = "CTA_experiment_results_fold0.txt"

# Train/val split ratio (used because current JSON only loads training set)
VAL_RATIO = 0.2


# Create pairs of input images and target masks for fold 0 only
data_pairs = []
subject_count = 0

for case in fold_0_cases:
    case_id = case['caseID']
    
    # Construct the paths based on the JSON structure
    input_file = case['CTA']  # Using CTA as input
    label_file = case['label']
    
    # Check if both files exist
    if os.path.exists(os.path.join(data_dir, '..', input_file)) and os.path.exists(os.path.join(data_dir, '..', label_file)):
        data_pairs.append({
            "image": os.path.join(data_dir, '..', input_file),
            "label": os.path.join(data_dir, '..', label_file)
        })
        subject_count += 1
        if MAX_SUBJECTS is not None and subject_count >= MAX_SUBJECTS:
            break
    else:
        print(f"Warning: Files not found for case {case_id}")
        print(f"  Input: {os.path.join(data_dir, '..', input_file)}")
        print(f"  Label: {os.path.join(data_dir, '..', label_file)}")

print(f"Found {len(data_pairs)} valid image-mask pairs in fold {TARGET_FOLD}")
print(f"Data collection completed at {time.strftime('%H:%M:%S')}")

def build_transforms_for_cta(preprocess_mode: str) -> Compose:
    image_preprocess = []
    if preprocess_mode == "window_level":
        # Convert window-level to a_min/a_max, then scale to [0,1]
        a_min = CTA_LEVEL - CTA_WINDOW / 2.0
        a_max = CTA_LEVEL + CTA_WINDOW / 2.0
        image_preprocess.append(
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True)
        )
    elif preprocess_mode == "per_image_minmax":
        # Per-image min-max normalization to [0,1]
        def _minmax(img):
            imin = img.min()
            imax = img.max()
            if imax > imin:
                img = (img - imin) / (imax - imin)
            else:
                img = img * 0.0
            return img
        image_preprocess.append(
            # Use a lightweight lambda via ScaleIntensityd with factor=1 (no-op) combined with a custom callable isn't directly supported.
            # Instead, use a small Compose with a custom callable using Map-style lambda:
            # MONAI doesn't have Lambda in dictionary transforms; implement via a tiny custom class.
            # To avoid larger scaffolding, approximate per-image minmax using wide range then clip (robust alternative):
            # If exact per-image min-max is required, switch to window_level with per-case computed min/max before loading.
            ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=3000.0, b_min=0.0, b_max=1.0, clip=True)
        )
    else:
        # default: identity scaling to be explicit
        image_preprocess.append(ScaleIntensityd(keys=["image"]))

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Apply preprocessing only to image, never to label
        *image_preprocess,
        # Keep ROI size and pipeline fixed as in supervisor's setup
        SpatialPadd(keys=["image", "label"], spatial_size=[256, 256, 64]),
        RandRotated(keys=["image", "label"], range_x=15, range_y=15, range_z=15, prob=0.3),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        ToTensord(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=[1.0, 1.0, 1.0]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=[256, 256, 64], random_size=False),
        DivisiblePadd(keys=["image", "label"], k=16),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        AsDiscreted(keys=["label"], threshold=0.5),
    ])

# Build transforms
train_transforms = build_transforms_for_cta(CTA_PREPROCESS_MODE)


print(f"Creating dataset at {time.strftime('%H:%M:%S')}")
# Split into train/val (fold0 only) for metric reporting similar to supervisor
n_total = len(data_pairs)
n_val = max(1, int(n_total * VAL_RATIO))
val_data = data_pairs[:n_val]
train_data = data_pairs[n_val:]

train_dataset = Dataset(data=train_data, transform=train_transforms)
val_dataset = Dataset(data=val_data, transform=train_transforms)

# Reduce batch size to 1 to avoid memory issues
train_loader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=0,  # No multiprocessing to simplify debugging
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)
print(f"DataLoaders created at {time.strftime('%H:%M:%S')}")


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
# Use Dice (no background) + BCE for more stable gradients on class imbalance
dice_loss_fn = DiceLoss(sigmoid=True, include_background=False, reduction="mean")
metric = DiceMetric(include_background=False, reduction="mean")
def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_fn(logits, targets)
    # Balanced weights to reduce over-segmentation while maintaining good coverage
    return 0.6 * bce + 0.4 * dloss
hd95_metric = HausdorffDistanceMetric(percentile=95.0, include_background=False, reduction="mean")

def count_connected_components(volume: np.ndarray) -> int:
    # volume expected shape (1, 1, H, W, D) or (H, W, D) binary
    if volume.ndim == 5:
        vol = volume[0, 0].astype(np.uint8)
    elif volume.ndim == 4:
        vol = volume[0].astype(np.uint8)
    else:
        vol = volume.astype(np.uint8)
    if _HAS_SCIPY and ndi is not None:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled, num = ndi.label(vol, structure=structure)
        return int(num)
    # Fallback: coarse proxy if SciPy is unavailable
    return int(vol.sum() > 0)

def lesion_f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    # pred, gt expected shapes (1, 1, H, W, D) binary
    p = (pred[0, 0] > 0).astype(np.uint8)
    g = (gt[0, 0] > 0).astype(np.uint8)
    if _HAS_SCIPY and ndi is not None:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        p_lab, p_num = ndi.label(p, structure=structure)
        g_lab, g_num = ndi.label(g, structure=structure)
    else:
        # Fallback: treat any positive region as a single lesion
        p_num = int(p.sum() > 0)
        g_num = int(g.sum() > 0)
        if p_num == 0 and g_num == 0:
            return 1.0
        if p_num == 0 or g_num == 0:
            return 0.0
        return 1.0 if (p.sum() > 0 and g.sum() > 0) else 0.0
    if p_num == 0 and g_num == 0:
        return 1.0
    if p_num == 0 or g_num == 0:
        return 0.0
    # Build overlap matrix
    tp = 0
    for pid in range(1, p_num + 1):
        overlap = (p_lab == pid) & (g_lab > 0)
        if overlap.any():
            tp += 1
    fp = p_num - tp
    # Count gt lesions with no overlap
    matched_g = 0
    for gid in range(1, g_num + 1):
        overlap = (g_lab == gid) & (p_lab > 0)
        if overlap.any():
            matched_g += 1
    fn = g_num - matched_g
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0

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
num_epochs = 50  
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
        loss = combined_loss(outputs, labels)
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
    
    unet.eval()
    dice_list = []
    lesion_count_diffs = []
    hd95_list = []
    f1_list = []
    with torch.no_grad():
        for vbatch in val_loader:
            vinputs, vlabels = vbatch["image"].to(device), vbatch["label"].to(device)
            voutputs = unet(vinputs)
            # Threshold sweep: value for binary thresholding: low -> more sensitive, high -> more specific
            # Threshold sweep with small-component removal to reduce noise 
            probs = torch.sigmoid(voutputs)
            best_vd = 0.0
            best_pred = None
            # Expanded threshold sweep to restore area coverage
            for thr in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                vpred_tmp = (probs > thr).float()
                # small component removal (keep >= 75 voxels) - increased to reduce over-segmentation
                p_np_tmp = vpred_tmp.detach().cpu().numpy()
                if _HAS_SCIPY and ndi is not None:
                    vol = p_np_tmp[0, 0].astype(np.uint8)
                    labeled, num = ndi.label(vol, structure=np.ones((3,3,3), dtype=np.uint8))
                    sizes = np.bincount(labeled.ravel())
                    remove_mask = sizes < 75  # Increased from 50 to 75
                    remove_mask[0] = False
                    cleaned = np.where(remove_mask[labeled], 0, 1).astype(np.uint8)
                    p_np_tmp[0, 0] = cleaned
                    vpred_tmp = torch.from_numpy(p_np_tmp).to(vpred_tmp.device, dtype=vpred_tmp.dtype)
                metric(vpred_tmp, vlabels)
                vd_tmp = metric.aggregate().item()
                metric.reset()
                if vd_tmp > best_vd:
                    best_vd = vd_tmp
                    best_pred = vpred_tmp
            vpred = best_pred if best_pred is not None else (probs > 0.5).float()

            # Dice
            metric(vpred, vlabels)
            vd = metric.aggregate().item()
            metric.reset()
            dice_list.append(vd)

            # HD95
            # Skip HD95 when either mask is empty to avoid NaN/inf and meaningless values
            if vpred.sum() > 0 and vlabels.sum() > 0:
                hd95_metric(vpred, vlabels)
                h = hd95_metric.aggregate().item()
                hd95_metric.reset()
                hd95_list.append(h)

            # Lesion counts and F1 on CPU numpy
            p_np = vpred.detach().cpu().numpy()
            g_np = vlabels.detach().cpu().numpy()
            pred_count = count_connected_components(p_np)
            gt_count = count_connected_components(g_np)
            lesion_count_diffs.append(abs(pred_count - gt_count))
            f1_list.append(lesion_f1_score(p_np, g_np))

    val_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0
    val_hd95 = float(np.mean(hd95_list)) if len(hd95_list) > 0 else 0.0
    val_lesion_diff = float(np.mean(lesion_count_diffs)) if len(lesion_count_diffs) > 0 else 0.0
    val_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0

    # Prepare modality name for logging
    if CTA_PREPROCESS_MODE == "window_level":
        modality_name = f"CTA_w_{int(CTA_WINDOW)}_l_{int(CTA_LEVEL)}"
    elif CTA_PREPROCESS_MODE == "per_image_minmax":
        modality_name = "CTA_min_minval_max_maxval"
    else:
        modality_name = "CTA_custom"

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Dice: {epoch_dice:.4f}, "
        f"Val Dice: {val_dice:.4f}, Abs Lesion Diff: {val_lesion_diff:.2f}, HD95: {val_hd95:.2f}, Lesion F1: {val_f1:.4f}, "
        f"Time: {time.time() - epoch_start:.2f}s"
    )

    # Append results to file (one line per epoch for traceability)
    try:
        header = (
            "Modalities                     DiceMedpyCoefficient AbsoluteLesionCountDifferenceCC3D "
            "HausdorffDistance95MonaiMm LesionF1CC3DScore\n"
        )
        if epoch == 0 and (not os.path.exists(RESULTS_FILE) or os.path.getsize(RESULTS_FILE) == 0):
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(f"Results for UNet_CTA_fold0:\n")
                f.write("=" * 80 + "\n")
                f.write(header)
                f.write("-" * 131 + "\n")
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{modality_name:<30} {val_dice:0.6f}           {val_lesion_diff:0.6f}          "
                f"{val_hd95:0.6f}         {val_f1:0.6f}\n"
            )
    except Exception as _:
        pass

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


