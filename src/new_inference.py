import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from model.DNANet_model import DNANet, Res_CBAM_block

# -----------------------------
# CONFIG (EDIT ONLY THESE)
# -----------------------------
IMAGE_DIR = "data/256/data 3/images"  # original images (1024x1280)
GT_DIR = "data/256/data 3/masks"  # original GT masks (1024x1280)
OUTPUT_DIR = "outputs/Final_checkpoint/data3"  # predictions (same size)
CHECKPOINT_PATH = "model/mIoU__DNANet_Final_Dataset_epoch.pth.tar"
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = DNANet(
    num_classes=1,
    input_channels=3,
    block=Res_CBAM_block,
    num_blocks=[2, 2, 2, 2],
    nb_filter=[16, 32, 64, 128, 256],
    deep_supervision=True,
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

print("DNANet loaded successfully")

# -----------------------------
# TRANSFORM (NO RESIZE)
# -----------------------------
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# -----------------------------
# IMAGE LIST
# -----------------------------
image_files = sorted(
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"Found {len(image_files)} images")

# -----------------------------
# INFERENCE (ORIGINAL SIZE)
# -----------------------------
with torch.no_grad():
    for fname in tqdm(image_files, desc="Running inference (original size)"):
        img_path = os.path.join(IMAGE_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)

        if not os.path.exists(gt_path):
            print(f"[SKIP] GT not found for {fname}")
            continue

        # --- Load original image (NO RESIZE)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # --- Forward pass
        outputs = model(img_tensor)
        pred = outputs[-1]

        # --- Logits → binary mask
        binary_mask = (pred > 0).float() * 255.0
        binary_mask = binary_mask[0, 0].cpu().numpy().astype(np.uint8)

        # --- Save (same size as input & GT)
        save_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(save_path, binary_mask)

print("✅ Inference completed. Predictions saved at original resolution.")
