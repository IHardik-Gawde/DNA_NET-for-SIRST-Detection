import os
import cv2
import matplotlib.pyplot as plt
import random

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "data/256/data 3/images"
GT_DIR = "data/256/data 3/masks"
PRED_DIR = "dataset_split/test/Final_Adam"

NUM_IMAGES = 3  # number of samples to visualize

# -----------------------------
# Collect image names
# -----------------------------
image_files = sorted(
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
)[:NUM_IMAGES]

all_images = [
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

image_files = random.sample(all_images, NUM_IMAGES)

print(f"Visualizing {len(image_files)} samples")

# -----------------------------
# Create figure
# -----------------------------
fig, axes = plt.subplots(
    nrows=len(image_files), ncols=3, figsize=(15, 4 * len(image_files)), squeeze=False
)

# Column titles
col_titles = ["Input Image", "Ground Truth", "Prediction"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=14, pad=12)

# -----------------------------
# Plot each sample
# -----------------------------
for i, fname in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, fname)
    gt_path = os.path.join(GT_DIR, fname)
    pred_path = os.path.join(PRED_DIR, fname)

    # --- Load input image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Load GT & prediction
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    pred = (
        cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(pred_path)
        else None
    )

    # -----------------
    # Input image
    # -----------------
    axes[i, 0].imshow(img)
    axes[i, 0].axis("off")

    # -----------------
    # Ground Truth
    # -----------------
    if gt is not None:
        axes[i, 1].imshow(gt, cmap="gray")
    else:
        axes[i, 1].text(0.5, 0.5, "GT not found", ha="center", va="center", fontsize=11)
    axes[i, 1].axis("off")

    # -----------------
    # Prediction
    # -----------------
    if pred is not None:
        axes[i, 2].imshow(pred, cmap="gray")
    else:
        axes[i, 2].text(
            0.5, 0.5, "Prediction not found", ha="center", va="center", fontsize=11
        )
    axes[i, 2].axis("off")

    # -----------------
    # Row title (filename)
    # -----------------
    fig.text(
        0.5,
        1 - (i + 0.5) / len(image_files),
        f"Sample: {fname}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
