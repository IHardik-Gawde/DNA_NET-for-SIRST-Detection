import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from NUAA_model import load_model
from NUAA import NUAADataset

# from model.BCE_loss import BCEDiceloss
from model.loss import SoftIoULoss
from model.metric import mIoU, ROCMetric


# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Load Model Architecture
# =========================
pretrained_path = "model/mIoU__DNANet_NUST-SIRST_epoch.pth.tar"
model = load_model(pretrained_path, device)

# Load BEST trained weights
model.load_state_dict(
    torch.load("best_model/Freezing/FinalAdam.pth", map_location=device)
)
model.to(device)
model.eval()

print("✅ Best model loaded")

# =========================
# BCE-LOSS
# =========================

# criterion = BCEDiceloss(bce_weight=0.5)


# =========================
# Dataset (same split as training)
# =========================
dataset_path = "dataset_split/test"
dataset = NUAADataset(dataset_path)

train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


# =========================
# Metrics
# =========================
metric_miou = mIoU(1)
metric_roc = ROCMetric(1, 10)


# =========================
# Evaluate Train Loss
# =========================
train_loss = 0

with torch.no_grad():
    for images, masks in tqdm(train_loader, desc="Evaluating Train"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        if isinstance(outputs, list):
            loss = 0
            for out in outputs:
                loss += SoftIoULoss(out, masks)
            loss = loss / len(outputs)
            pred = outputs[-1]
        else:
            loss = SoftIoULoss(outputs, masks)
            pred = outputs

        train_loss += loss.item()

train_loss /= len(train_loader)


# =========================
# Evaluate Validation
# =========================
val_loss = 0
metric_miou.reset()
metric_roc.reset()

with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Evaluating Validation"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        if isinstance(outputs, list):
            loss = 0
            for out in outputs:
                loss += SoftIoULoss(out, masks)
            loss = loss / len(outputs)
            pred = outputs[-1]
        else:
            loss = SoftIoULoss(outputs, masks)
            pred = outputs

        val_loss += loss.item()

        metric_miou.update(pred, masks)
        metric_roc.update(pred, masks)

val_loss /= len(val_loader)

pixAcc, meanIoU = metric_miou.get()
tpr, fpr, recall, precision = metric_roc.get()


# =========================
# Print Final Results
# =========================
print("\n========== BEST MODEL EVALUATION ==========")
print("Train Loss:", train_loss)
print("Validation Loss:", val_loss)
print("Mean IoU:", meanIoU)
print("PD @0.8:", tpr[8])
print("FA @0.8:", fpr[8])
