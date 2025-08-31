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
import json
import time
from typing import List, Tuple

import numpy as np
import torch

from monai.config import print_config
from monai.utils import set_determinism, InterpolateMode
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.metrics import HausdorffDistanceMetric, DiceMetric
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    ScaleIntensityRanged,
    ToTensord,
    AsDiscreted,
)

try:
    from medpy.metric.binary import dc as medpy_dice
    _HAS_MEDPY = True
except Exception:
    _HAS_MEDPY = False

try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    ndi = None
    _HAS_SCIPY = False


# CONFIG

CONFIG = {
    "DATA_DIR": r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES",
    "JSON_FILE": r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\isles24_multimodal_5fold_NIHSSstratified.json",

    "TARGET_FOLD": 0,


    "MODALITY": "CTA",
    "PREPROCESS_MODE": "window_level",
    "CTA_WINDOW": 75,
    "CTA_LEVEL": 40,

    # Training
    "NUM_EPOCHS": 120,
    "BATCH_SIZE": 1,
    "NUM_WORKERS": 0,
    "LR": 2e-4,
    "WEIGHT_DECAY": 1e-5,
    "WARMUP_EPOCHS": 10,

    # ROI / inference
    "ROI_SIZE": (128, 128, 64),
    "SW_OVERLAP": 0.5,

    # Postprocessing & thresholding
    "MIN_COMPONENT_SIZE": 600,  # Voxel (1mm^3)
    # Fester globaler Threshold, wird jetzt immer verwendet.
    "GLOBAL_THRESHOLD": 0.5,

    # Results
    "RESULTS_FILE": "CTA_experiment_results_fold0.txt",

    # Reproducibility
    "SEED": 42,
}


print_config()


set_determinism(seed=CONFIG["SEED"]) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data loading helpers

def load_fold_cases(json_file: str, target_fold: int):
    with open(json_file, "r") as f:
        fold_data = json.load(f)
    return [case for case in fold_data["training"] if case["fold"] == target_fold]


def build_datalist(fold_cases, data_dir: str, modality_key: str):
    data_pairs = []
    base = os.path.join(data_dir, "..")  # JSON paths sind relativ eine Ebene höher
    for case in fold_cases:
        image_rel = case.get(modality_key)
        label_rel = case.get("label")
        if not image_rel or not label_rel:
            continue
        image_path = os.path.join(base, image_rel)
        label_path = os.path.join(base, label_rel)
        if os.path.exists(image_path) and os.path.exists(label_path):
            data_pairs.append({"image": image_path, "label": label_path})
        else:
            print("[WARN] Missing files:", image_path, label_path)
    return data_pairs



# Intensity preprocessing
def intensity_transform_for_cta(preprocess_mode: str, window: float, level: float):
    if preprocess_mode == "window_level":
        a_min = level - window / 2.0
        a_max = level + window / 2.0
        return ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True)
    # fallback broad minmax (nicht optimal für CTA)
    return ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=3000.0, b_min=0.0, b_max=1.0, clip=True)



# Transforms
def get_train_transforms(cfg) -> Compose:
    intensity_t = intensity_transform_for_cta(cfg["PREPROCESS_MODE"], cfg["CTA_WINDOW"], cfg["CTA_LEVEL"])
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=(InterpolateMode.BILINEAR, InterpolateMode.NEAREST)),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        intensity_t,
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg["ROI_SIZE"],
            pos=2, neg=1, num_samples=2,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandRotated(keys=["image", "label"], range_x=0.26, range_y=0.26, range_z=0.26, prob=0.3,
                    mode=(InterpolateMode.BILINEAR, InterpolateMode.NEAREST)),
        AsDiscreted(keys=["label"], threshold=0.5),  # Labels am Ende härten
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(cfg) -> Compose:
    intensity_t = intensity_transform_for_cta(cfg["PREPROCESS_MODE"], cfg["CTA_WINDOW"], cfg["CTA_LEVEL"])
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=(InterpolateMode.BILINEAR, InterpolateMode.NEAREST)),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        intensity_t,
        AsDiscreted(keys=["label"], threshold=0.5),
        ToTensord(keys=["image", "label"]),
    ])



# Postprocessing & metric helpers
def clean_mask(vpred_np: np.ndarray, min_size: int = 600) -> np.ndarray:
    """Remove tiny components, de-noise (opening), and smooth (closing).
    Expects shape (1,1,H,W,D), values in {0,1}.
    """
    assert vpred_np.ndim == 5 and vpred_np.shape[0] == 1 and vpred_np.shape[1] == 1, "Expect (1,1,H,W,D)"
    if not (_HAS_SCIPY and ndi is not None):
        return vpred_np
    vol = vpred_np[0, 0].astype(np.uint8)

    # 1) Remove small components
    if min_size > 0:
        labeled, num = ndi.label(vol, structure=np.ones((3, 3, 3), dtype=np.uint8))
        if num > 0:
            sizes = np.bincount(labeled.ravel())
            keep = sizes >= min_size
            keep[0] = False
            vol = np.where(keep[labeled], 1, 0).astype(np.uint8)
        else:
            vol = np.zeros_like(vol)


    # 2) Morphological opening (remove small false detections)
    vol = ndi.binary_opening(vol, structure=np.ones((3, 3, 3), dtype=bool), iterations=1).astype(np.uint8)

    # 3) Morphological closing (fill small holes / smooth)
    vol = ndi.binary_closing(vol, structure=np.ones((3, 3, 3), dtype=bool), iterations=1).astype(np.uint8)

    vpred_np[0, 0] = vol
    return vpred_np


def count_connected_components(volume: np.ndarray) -> int:
    # volume can be (1,1,H,W,D) or (H,W,D)
    if volume.ndim == 5:
        vol = volume[0, 0].astype(np.uint8)
    elif volume.ndim == 3:
        vol = volume.astype(np.uint8)
    else:
        raise ValueError("Unsupported volume shape for CC count")
    if _HAS_SCIPY and ndi is not None:
        _, num = ndi.label(vol, structure=np.ones((3, 3, 3), dtype=np.uint8))
        return int(num)
    return int(vol.sum() > 0)


def lesion_f1_cc3d(pred: np.ndarray, gt: np.ndarray) -> float:
    # pred, gt shapes: (1,1,H,W,D) oder (H,W,D)
    if pred.ndim == 5:
        p = (pred[0, 0] > 0).astype(np.uint8)
    else:
        p = (pred > 0).astype(np.uint8)
    if gt.ndim == 5:
        g = (gt[0, 0] > 0).astype(np.uint8)
    else:
        g = (gt > 0).astype(np.uint8)
    if _HAS_SCIPY and ndi is not None:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        p_lab, p_num = ndi.label(p, structure=structure)
        g_lab, g_num = ndi.label(g, structure=structure)
    else:
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
    tp = 0
    for pid in range(1, p_num + 1):
        if np.any((p_lab == pid) & (g_lab > 0)):
            tp += 1
    fp = p_num - tp
    matched_g = 0
    for gid in range(1, g_num + 1):
        if np.any((g_lab == gid) & (p_lab > 0)):
            matched_g += 1
    fn = g_num - matched_g
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0



# Build datasets & loaders

print(f"Loading fold {CONFIG['TARGET_FOLD']} cases from {CONFIG['JSON_FILE']}")
fold_cases = load_fold_cases(CONFIG["JSON_FILE"], CONFIG["TARGET_FOLD"])
print(f"Found {len(fold_cases)} cases in fold {CONFIG['TARGET_FOLD']}")

datalist = build_datalist(fold_cases, CONFIG["DATA_DIR"], CONFIG["MODALITY"])
print(f"Valid pairs: {len(datalist)}")

VAL_RATIO = 0.2
n_total = len(datalist)
assert n_total > 1, "Not enough cases in this fold."

n_val = max(1, int(n_total * VAL_RATIO))
val_files = datalist[:n_val]
train_files = datalist[n_val:]
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

train_transforms = get_train_transforms(CONFIG)
val_transforms = get_val_transforms(CONFIG)

train_ds = Dataset(train_files, transform=train_transforms)
val_ds = Dataset(val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=CONFIG["NUM_WORKERS"], pin_memory=torch.cuda.is_available())


# Model, loss, optimizer, scheduler

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,  # residual blocks
    norm="INSTANCE",
).to(device)

loss_fn = DiceCELoss(sigmoid=True, include_background=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])

# Warmup + Cosine schedule
warmup_epochs = CONFIG["WARMUP_EPOCHS"]
num_epochs = CONFIG["NUM_EPOCHS"]

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Metrics / inferer
hd95_metric = HausdorffDistanceMetric(percentile=95.0, include_background=False, reduction="mean")
patch_dice = DiceMetric(include_background=False, reduction="mean")  # nur fürs Training-Log
inferer = SlidingWindowInferer(roi_size=CONFIG["ROI_SIZE"], sw_batch_size=1, overlap=CONFIG["SW_OVERLAP"])


# Results header

modality_name = (
    f"CTA_w_{int(CONFIG['CTA_WINDOW'])}_l_{int(CONFIG['CTA_LEVEL'])}" if CONFIG["PREPROCESS_MODE"] == "window_level" else
    f"CTA_min_minval_max_maxval"
)

header = (
    "Modalities                     DiceMedpyCoefficient AbsoluteLesionCountDifferenceCC3D "
    "HausdorffDistance95MonaiMm LesionF1CC3DScore"
)
results_file_path = CONFIG["RESULTS_FILE"]
if not os.path.exists(results_file_path) or os.path.getsize(results_file_path) == 0:
    with open(results_file_path, "a", encoding="utf-8") as f:
        f.write(f"Results for UNet_{CONFIG['MODALITY']}_fold{CONFIG['TARGET_FOLD']}:\n")
        f.write("=" * 80 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")


# Training & Validation

print("Starting training...")
start_time_all = time.time()

if not _HAS_MEDPY:
    raise ImportError("medpy is required for DiceMedpyCoefficient. Please install via `pip install medpy`.")

best_val_dice = -1.0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    patch_dice.reset()
    t0 = time.time()

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

        
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            patch_dice(preds, labels)

    train_dice = patch_dice.aggregate().item()
    patch_dice.reset()
    epoch_loss /= max(1, len(train_loader))

    #Validation auf ganzen Volumina
    model.eval()

    dice_medpy_list, lesion_diff_list, hd95_list, f1_list = [], [], [], []
    global_thr = float(CONFIG["GLOBAL_THRESHOLD"])

    with torch.no_grad():
        for vbatch in val_loader:
            vimg = vbatch["image"].to(device)
            vlab = vbatch["label"].cpu().numpy()[0, 0].astype(np.uint8) # (H,W,D)
            
            logits = inferer(vimg, model)
            probs = torch.sigmoid(logits).cpu().numpy() # (1,1,H,W,D)

            pred_bin = (probs > global_thr).astype(np.uint8)
            pred_bin = clean_mask(pred_bin, CONFIG["MIN_COMPONENT_SIZE"])  # (1,1,H,W,D)
            pp = pred_bin[0, 0]

            # 1) Dice (medpy)
            d = float(medpy_dice(pp, vlab)) if (pp.sum() > 0 or vlab.sum() > 0) else 1.0
            dice_medpy_list.append(d)

            # 2) Abs Lesion Count Diff
            pred_count = count_connected_components(pp)
            gt_count = count_connected_components(vlab)
            lesion_diff_list.append(abs(pred_count - gt_count))

            # 3) HD95 (95th percentile of the Hausdorff distance)
            if pp.sum() > 0 and vlab.sum() > 0:
                # Convert back to tensors for metric calculation
                pp_t = torch.from_numpy(pred_bin).to(device=device, dtype=torch.float32)
                gt_t = torch.from_numpy(vlab[None, None]).to(device=device, dtype=torch.float32)
                hd95_metric(pp_t, gt_t)
                h = float(hd95_metric.aggregate().item())
                hd95_metric.reset()
                hd95_list.append(h)
            else:
                # If one of the masks is empty, calculate the maximum distance (diagonal) as a penalty value
                shape = vlab.shape
                max_dist = np.sqrt(shape[0]**2 + shape[1]**2 + shape[2]**2)
                hd95_list.append(max_dist)

            # 4) Lesion F1 (CC3D)
            f1_list.append(lesion_f1_cc3d(pred_bin, vlab))

    # Aggregated metrics
    DiceMedpyCoefficient = float(np.mean(dice_medpy_list)) if dice_medpy_list else 0.0
    AbsoluteLesionCountDifferenceCC3D = float(np.mean(lesion_diff_list)) if lesion_diff_list else 0.0
    HausdorffDistance95MonaiMm = float(np.mean(hd95_list)) if hd95_list else 0.0 # Sollte dank Strafwert nie 0 sein
    LesionF1CC3DScore = float(np.mean(f1_list)) if f1_list else 0.0

    # LR step
    scheduler.step()

    # Console log (Train and metrics)
    elapsed = time.time() - t0
    print(
        f"Epoch {epoch+1}/{num_epochs} | Loss {epoch_loss:.4f} | Train Dice {train_dice:.4f} | "
        f"Val DiceMedpy {DiceMedpyCoefficient:.4f} | Abs Lesion Diff {AbsoluteLesionCountDifferenceCC3D:.2f} | "
        f"HD95 {HausdorffDistance95MonaiMm:.2f} | Lesion F1 {LesionF1CC3DScore:.4f} | Thr {global_thr:.2f} | "
        f"LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
    )

    try:
        with open(results_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{modality_name:<30} {DiceMedpyCoefficient:0.6f}           {AbsoluteLesionCountDifferenceCC3D:0.6f}          "
                f"{HausdorffDistance95MonaiMm:0.6f}          {LesionF1CC3DScore:0.6f}\n"
            )
    except Exception as e:
        print(f"[WARN] could not write results: {e}")

    # Save best model based on validation Dice
    if DiceMedpyCoefficient > best_val_dice:
        best_val_dice = DiceMedpyCoefficient
        ckpt_name = f"unet_{CONFIG['MODALITY'].lower()}_fold{CONFIG['TARGET_FOLD']}_best.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"New best model saved to {ckpt_name} (Val Dice: {best_val_dice:.4f})")


# Save final model
ckpt_name = f"unet_{CONFIG['MODALITY'].lower()}_fold{CONFIG['TARGET_FOLD']}_final.pth"
torch.save(model.state_dict(), ckpt_name)
print(f"Saved final model to {ckpt_name}")
print(f"Total run time: {(time.time() - start_time_all) / 60:.2f} minutes")