import torch
import time
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler

from NUAA_model import load_model
from NUAA import NUAADataset

# from model.BCE_loss import BCEDiceloss
from model.loss import SoftIoULoss

from model.metric import mIoU, ROCMetric


# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

# =========================
# Load Pretrained NUAA Model
# =========================
pretrained_path = "model/mIoU__DNANet_NUST-SIRST_epoch.pth.tar"
model = load_model(pretrained_path, device)

# =========================
# Freeze Encoder
# =========================
for name, param in model.named_parameters():
    if "conv0_0" in name or "conv1_0" in name or "conv2_0" in name or "conv3_0" in name:
        param.requires_grad = False
    # =========================
    # # Freeze Decoder
    # =========================

    # if (
    #     "conv0_1" in name
    #     or "conv1_1" in name
    #     or "conv2_1" in name
    #     or "conv3_1" in name
    #     or "conv0_2" in name
    # ):
    #     param.requires_grad = False


# =========================
# Dataset
# =========================
train_dataset = NUAADataset("dataset_split/train")
val_dataset = NUAADataset("dataset_split/val")


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# =========================
# Optimizer & Scheduler
# =========================
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
# criterion = BCEDiceloss(bce_weight=0.5)

# =========================
# Metrics
# =========================
metric_miou = mIoU(1)
metric_roc = ROCMetric(1, 10)

# =========================
# Training
# =========================
epochs = 50
best_miou = 0
total_start_time = time.time()


for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"\n====== Epoch {epoch + 1}/{epochs} ======")

    # ---------------------
    # Train
    # ---------------------
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # Deep supervision handling
        if isinstance(outputs, list):
            loss = 0
            for out in outputs:
                loss += SoftIoULoss(out, masks)
            loss = loss / len(outputs)
        else:
            loss = SoftIoULoss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    print("Train Loss:", train_loss / len(train_loader))

    # ---------------------
    # Validation
    # ---------------------
    model.eval()
    metric_miou.reset()
    metric_roc.reset()
    val_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
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

    pixAcc, meanIoU = metric_miou.get()
    tpr, fpr, recall, precision = metric_roc.get()

    print("Validation Loss:", val_loss / len(val_loader))
    print("Validation mIoU:", meanIoU)

    # Save best model
    if meanIoU > best_miou:
        best_miou = meanIoU

        os.makedirs("best_model/Freezing", exist_ok=True)
        torch.save(model.state_dict(), "best_model/Freezing/FinalAdam.pth")
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch Time: {epoch_time:.2f} seconds")

    torch.cuda.empty_cache()

total_training_time = time.time() - total_start_time

print("\n Training Completed")
print("Best Validation mIoU:", best_miou)
print(f" Total Training Time: {total_training_time:.2f} seconds")
print(f" Total Training Time: {total_training_time / 60:.2f} minutes")
print(f" Total Training Time: {total_training_time / 3600:.2f} hours")
