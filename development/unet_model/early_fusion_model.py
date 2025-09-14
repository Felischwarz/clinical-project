import os
import json
import time
from typing import List, Dict

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
    SpatialPadd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    ScaleIntensityRanged,
    ConcatItemsd, #used for fusion
    AsDiscreted,
)

try:
    from medpy.metric.binary import dc as medpy_dice
    _HAS_MEDPY = True
except ImportError:
    _HAS_MEDPY = False

try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except ImportError:
    ndi = None
    _HAS_SCIPY = False


# CONFIG

CONFIG = {
    "DATA_DIR": r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\DERIVATIVES",
    "JSON_FILE": r"D:\TUM_CLINICAL_PROJECT\ISLES24_COMBINED\isles24_multimodal_5fold_NIHSSstratified.json",

    "TARGET_FOLD": 1,  # Train on any fold from 0 to 4
    "MODALITIES": ["cbf", "tmax"],  

    
    "PERFUSION_A_MIN": 0.0,
    "PERFUSION_A_MAX": 200.0, 


    "NUM_EPOCHS": 120,
    "BATCH_SIZE": 1,
    "NUM_WORKERS": 8,
    "LR": 2e-4,
    "WEIGHT_DECAY": 1e-5,
    "WARMUP_EPOCHS": 10,

    "ROI_SIZE": (128, 128, 64),
    "SW_OVERLAP": 0.5,

    "MIN_COMPONENT_SIZE": 600,
    "GLOBAL_THRESHOLD": 0.5,


    "RESULTS_FILE": "multimodal_experiment_results.txt",

    "SEED": 42,
}


print_config()
set_determinism(seed=CONFIG["SEED"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data loading helpers
def load_fold_cases(json_file: str, target_fold: int):
    
    with open(json_file, "r") as f:
        all_data = json.load(f)
    
    train_cases = [case for case in all_data["training"] if case["fold"] != target_fold]
    val_cases = [case for case in all_data["training"] if case["fold"] == target_fold]
    
    return train_cases, val_cases


def build_datalist(cases: List[Dict], data_dir: str, modality_keys: List[str]) -> List[Dict]:
    data_list = []
    base = os.path.join(data_dir, "..") 
    for case in cases:
        item = {}
        all_files_exist = True
        
        for key in modality_keys:
            rel_path = case.get(key.upper())

            if not rel_path:
                all_files_exist = False
                break
            full_path = os.path.join(base, rel_path)
            if os.path.exists(full_path):
                item[key] = full_path
            else:
                print(f"Missing file for case {case.get('caseID', case.get('case', 'N/A'))}: {full_path}")
                all_files_exist = False
                break
        
        if not all_files_exist:
            continue

       
        label_rel = case.get("label")
        if not label_rel:
            continue
        label_path = os.path.join(base, label_rel)
        if os.path.exists(label_path):
            item["label"] = label_path
        else:
            print(f"Missing label for case {case.get('caseID', case.get('case', 'N/A'))}: {label_path}")
            continue

        data_list.append(item)
        
    return data_list


def get_transforms(cfg, mode: str) -> Compose:
    image_keys = cfg["MODALITIES"]
    all_keys = image_keys + ["label"]

    intensity_transforms = [
        ScaleIntensityRanged(
            keys=key, a_min=cfg["PERFUSION_A_MIN"], a_max=cfg["PERFUSION_A_MAX"],
            b_min=0.0, b_max=1.0, clip=True
        ) for key in image_keys
    ]

    # Base transforms applied to all data
    transforms = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                 mode=([InterpolateMode.BILINEAR] * len(image_keys)) + [InterpolateMode.NEAREST]),
        CropForegroundd(keys=all_keys, source_key=image_keys[0], allow_smaller=True), 
        # Pad AFTER foreground crop to ensure final volume is >= ROI along all dims
        SpatialPadd(keys=all_keys, spatial_size=cfg["ROI_SIZE"], mode="edge"),
        *intensity_transforms,
        #FUSION STEP
        ConcatItemsd(keys=image_keys, name="image", dim=0),
    ]

    if mode == "train":
        transforms.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=cfg["ROI_SIZE"], pos=2, neg=1,
                num_samples=2, image_key="image", allow_smaller=True
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotated(keys=["image", "label"], range_x=0.26, range_y=0.26, range_z=0.26, prob=0.3,
                        mode=(InterpolateMode.BILINEAR, InterpolateMode.NEAREST)),
        ])

    transforms.extend([
        AsDiscreted(keys=["label"], threshold=0.5),
        ToTensord(keys=["image", "label"]),
    ])

    return Compose(transforms)

# these functions are the same as in main.py
def clean_mask(vpred_np: np.ndarray, min_size: int = 600) -> np.ndarray:
    assert vpred_np.ndim == 5 and vpred_np.shape[0] == 1 and vpred_np.shape[1] == 1, "Expect (1,1,H,W,D)"
    if not (_HAS_SCIPY and ndi is not None):
        return vpred_np
    vol = vpred_np[0, 0].astype(np.uint8)
    if min_size > 0:
        labeled, num = ndi.label(vol, structure=np.ones((3, 3, 3), dtype=np.uint8))
        if num > 0:
            sizes = np.bincount(labeled.ravel())
            keep = sizes >= min_size
            keep[0] = False
            vol = np.where(keep[labeled], 1, 0).astype(np.uint8)
        else:
            vol = np.zeros_like(vol)
    vol = ndi.binary_opening(vol, structure=np.ones((3, 3, 3), dtype=bool), iterations=1).astype(np.uint8)
    vol = ndi.binary_closing(vol, structure=np.ones((3, 3, 3), dtype=bool), iterations=1).astype(np.uint8)
    vpred_np[0, 0] = vol
    return vpred_np

def count_connected_components(volume: np.ndarray) -> int:
    if volume.ndim == 5: vol = volume[0, 0].astype(np.uint8)
    elif volume.ndim == 3: vol = volume.astype(np.uint8)
    else: raise ValueError("Unsupported volume shape for CC count")
    if _HAS_SCIPY and ndi is not None:
        _, num = ndi.label(vol, structure=np.ones((3, 3, 3), dtype=np.uint8))
        return int(num)
    return int(vol.sum() > 0)

def lesion_f1_cc3d(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.ndim == 5: p = (pred[0, 0] > 0).astype(np.uint8)
    else: p = (pred > 0).astype(np.uint8)
    if gt.ndim == 5: g = (gt[0, 0] > 0).astype(np.uint8)
    else: g = (gt > 0).astype(np.uint8)
    if not (_HAS_SCIPY and ndi is not None):
        p_num = int(p.sum() > 0)
        g_num = int(g.sum() > 0)
        if p_num == 0 and g_num == 0: return 1.0
        if p_num == 0 or g_num == 0: return 0.0
        return 1.0 if (p.sum() > 0 and g.sum() > 0) else 0.0
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    p_lab, p_num = ndi.label(p, structure=structure)
    g_lab, g_num = ndi.label(g, structure=structure)
    if p_num == 0 and g_num == 0: return 1.0
    if p_num == 0 or g_num == 0: return 0.0
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
print(f"Loading data for fold {CONFIG['TARGET_FOLD']} as validation set.")
train_cases, val_cases = load_fold_cases(CONFIG["JSON_FILE"], CONFIG["TARGET_FOLD"])
print(f"Total train cases: {len(train_cases)}, Total val cases: {len(val_cases)}")

train_files = build_datalist(train_cases, CONFIG["DATA_DIR"], CONFIG["MODALITIES"])
val_files = build_datalist(val_cases, CONFIG["DATA_DIR"], CONFIG["MODALITIES"])
print(f"Found {len(train_files)} valid training pairs and {len(val_files)} valid validation pairs.")
assert len(train_files) > 0 and len(val_files) > 0, "Not enough data for training or validation."

train_transforms = get_transforms(CONFIG, mode="train")
val_transforms = get_transforms(CONFIG, mode="val")

train_ds = Dataset(train_files, transform=train_transforms)
val_ds = Dataset(val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=CONFIG["NUM_WORKERS"], pin_memory=torch.cuda.is_available())

# Model, loss, optimizer, scheduler
model = UNet(
    spatial_dims=3,
    in_channels=len(CONFIG["MODALITIES"]), #Key model change for fusion
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
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
patch_dice = DiceMetric(include_background=False, reduction="mean")
inferer = SlidingWindowInferer(roi_size=CONFIG["ROI_SIZE"], sw_batch_size=1, overlap=CONFIG["SW_OVERLAP"])



modality_name = "_".join(CONFIG["MODALITIES"]).upper()
header = (
    "Modalities                     DiceMedpyCoefficient AbsoluteLesionCountDifferenceCC3D "
    "HausdorffDistance95MonaiMm LesionF1CC3DScore"
)
results_file_path = CONFIG["RESULTS_FILE"]
if not os.path.exists(results_file_path) or os.path.getsize(results_file_path) == 0:
    with open(results_file_path, "a", encoding="utf-8") as f:
        f.write(f"Results for UNet_multimodal_experiments:\n")
        f.write("=" * 120 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

if __name__ == "__main__":
    print("Starting training...")
    start_time_all = time.time()
    if not _HAS_MEDPY:
        raise ImportError("medpy is required.")
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

        
        model.eval()
        dice_medpy_list, lesion_diff_list, hd95_list, f1_list = [], [], [], []
        global_thr = float(CONFIG["GLOBAL_THRESHOLD"])

        with torch.no_grad():
            for vbatch in val_loader:
                vimg = vbatch["image"].to(device)
                vlab = vbatch["label"].cpu().numpy()[0, 0].astype(np.uint8)
                
                logits = inferer(vimg, model)
                probs = torch.sigmoid(logits).cpu().numpy()

                pred_bin = (probs > global_thr).astype(np.uint8)
                pred_bin = clean_mask(pred_bin, CONFIG["MIN_COMPONENT_SIZE"])
                pp = pred_bin[0, 0]

                d = float(medpy_dice(pp, vlab)) if (pp.sum() > 0 or vlab.sum() > 0) else 1.0
                dice_medpy_list.append(d)

                pred_count = count_connected_components(pp)
                gt_count = count_connected_components(vlab)
                lesion_diff_list.append(abs(pred_count - gt_count))

                if pp.sum() > 0 and vlab.sum() > 0:
                    pp_t = torch.from_numpy(pred_bin).to(device=device, dtype=torch.float32)
                    gt_t = torch.from_numpy(vlab[None, None]).to(device=device, dtype=torch.float32)
                    hd95_metric(pp_t, gt_t)
                    h = float(hd95_metric.aggregate().item())
                    hd95_metric.reset()
                    hd95_list.append(h)
                elif vlab.sum() > 0:
                    shape = vlab.shape
                    max_dist = np.sqrt(shape[0]**2 + shape[1]**2 + shape[2]**2)
                    hd95_list.append(max_dist)
                # If both are empty, HD95 is 0, which is correct.
                
                f1_list.append(lesion_f1_cc3d(pred_bin, vlab))

        DiceMedpyCoefficient = float(np.mean(dice_medpy_list)) if dice_medpy_list else 0.0
        AbsoluteLesionCountDifferenceCC3D = float(np.mean(lesion_diff_list)) if lesion_diff_list else 0.0
        HausdorffDistance95MonaiMm = float(np.mean(hd95_list)) if hd95_list else 0.0
        LesionF1CC3DScore = float(np.mean(f1_list)) if f1_list else 0.0

        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss {epoch_loss:.4f} | Train Dice {train_dice:.4f} | "
            f"Val DiceMedpy {DiceMedpyCoefficient:.4f} | Abs Lesion Diff {AbsoluteLesionCountDifferenceCC3D:.2f} | "
            f"HD95 {HausdorffDistance95MonaiMm:.2f} | Lesion F1 {LesionF1CC3DScore:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        try:
            with open(results_file_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{modality_name}_F{CONFIG['TARGET_FOLD']:<29} "
                    f"{DiceMedpyCoefficient:0.6f} "
                    f"{AbsoluteLesionCountDifferenceCC3D:0.6f} "
                    f"{HausdorffDistance95MonaiMm:0.6f} "
                    f"{LesionF1CC3DScore:0.6f}\n"
                )
        except Exception as e:
            print(f"[WARN] could not write results: {e}")

        if DiceMedpyCoefficient > best_val_dice:
            best_val_dice = DiceMedpyCoefficient
            ckpt_name = f"unet_{modality_name.lower()}_fold{CONFIG['TARGET_FOLD']}_best.pth"
            torch.save(model.state_dict(), ckpt_name)
            print(f"New best model saved to {ckpt_name} (Val Dice: {best_val_dice:.4f})")


    ckpt_name = f"unet_{modality_name.lower()}_fold{CONFIG['TARGET_FOLD']}_final.pth"
    torch.save(model.state_dict(), ckpt_name)
    print(f"Saved final model to {ckpt_name}")
    print(f"Total run time: {(time.time() - start_time_all) / 60:.2f} minutes")