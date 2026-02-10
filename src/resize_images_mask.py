import os
import cv2

# ==============================
# EDIT THESE PATHS
# ==============================
IMG_DIR = "data/data 3/images"  # 1024x1280 images
GT_DIR = "data/data 3/masks"  # 1024x1280 masks

OUT_IMG_DIR = "data/256/data 3/images"
OUT_GT_DIR = "data/256/data 3/masks"

TARGET_SIZE = (256, 256)  # (width, height)
# ==============================


os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_GT_DIR, exist_ok=True)

valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith(valid_exts):
        continue

    img_path = os.path.join(IMG_DIR, fname)
    gt_path = os.path.join(GT_DIR, fname)

    if not os.path.exists(gt_path):
        print(f"[SKIP] GT not found for {fname}")
        continue

    # -----------------------------
    # Load image & mask
    # -----------------------------
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt is None:
        print(f"[SKIP] Cannot read {fname}")
        continue

    # -----------------------------
    # Resize
    # -----------------------------
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    gt_resized = cv2.resize(gt, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # -----------------------------
    # Save
    # -----------------------------
    cv2.imwrite(os.path.join(OUT_IMG_DIR, fname), img_resized)
    cv2.imwrite(os.path.join(OUT_GT_DIR, fname), gt_resized)

print("✅ Images and masks resized to 256×256 successfully.")
