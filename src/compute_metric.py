import os
import cv2
import numpy as np

# ==============================
# EDIT ONLY THESE PATHS
# ==============================
PRED_DIR = "outputs/Final_checkpoint/data3"  # resized predictions (logits or binary)
GT_DIR = "data/256/data 3/masks"  # ground truth masks
# ==============================


# ------------------------------
# Helpers
# ------------------------------
def binarize_pred(pred):
    # logit-based threshold
    return (pred > 0).astype(np.uint8)


def binarize_gt(gt):
    return (gt > 0).astype(np.uint8)


def compute_iou(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return inter / (union + 1e-8)


def compute_pd(pred_bin, gt_bin):
    num_gt, gt_labels = cv2.connectedComponents(gt_bin)
    detected = 0

    for i in range(1, num_gt):  # skip background
        gt_comp = gt_labels == i
        if np.any(pred_bin[gt_comp]):
            detected += 1

    return detected, max(num_gt - 1, 1)


def compute_fa(pred_bin, gt_bin):
    num_pred, pred_labels = cv2.connectedComponents(pred_bin)
    fa = 0

    for i in range(1, num_pred):  # skip background
        pred_comp = pred_labels == i
        if not np.any(gt_bin[pred_comp]):
            fa += 1

    return fa, num_pred - 1


# ------------------------------
# Main evaluation loop
# ------------------------------
ious = []
total_gt_targets = 0
total_gt_detected = 0
total_fa = 0
total_pred_components = 0

images_evaluated = 0
gt_target_images = 0
background_only_images = 0

valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for fname in sorted(os.listdir(PRED_DIR)):
    if not fname.lower().endswith(valid_exts):
        continue

    pred_path = os.path.join(PRED_DIR, fname)
    gt_path = os.path.join(GT_DIR, fname)

    if not os.path.exists(gt_path):
        continue

    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        continue

    if pred.ndim == 3:
        pred = pred[:, :, 0]

    pred_bin = binarize_pred(pred)
    gt_bin = binarize_gt(gt)

    images_evaluated += 1

    if np.any(gt_bin):
        gt_target_images += 1
    else:
        background_only_images += 1

    # IoU
    ious.append(compute_iou(pred_bin, gt_bin))

    # PD
    detected, gt_count = compute_pd(pred_bin, gt_bin)
    total_gt_detected += detected
    total_gt_targets += gt_count

    # FA (CORRECTED)
    fa, pred_count = compute_fa(pred_bin, gt_bin)
    total_fa += fa
    total_pred_components += pred_count


# ------------------------------
# Final metrics
# ------------------------------
mean_iou = np.mean(ious) if ious else 0
pd = total_gt_detected / (total_gt_targets + 1e-8)
fa = total_fa / max(images_evaluated, 1)  # FA per image

# ------------------------------
# Report
# ------------------------------
print("\n================ METRICS ================")
print(f"Images evaluated        : {images_evaluated}")
print(f"GT target images        : {gt_target_images}")
print(f"Background-only images  : {background_only_images}")
print("----------------------------------------")
print(f"Mean IoU                : {mean_iou:.4f}")
print(f"Pd (Probability detect) : {pd:.4f}")
print(f"Fa (False alarm rate)   : {fa:.4f}")
print("=========================================")

print("\n[Debug]")
print("Total GT targets        :", total_gt_targets)
print("Total detected targets :", total_gt_detected)
print("Total pred components  :", total_pred_components)
print("Total FA components    :", total_fa)
