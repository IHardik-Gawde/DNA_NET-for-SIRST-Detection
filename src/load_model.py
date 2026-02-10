import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model.DNANet_model import DNANet, Res_CBAM_block

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)

# load model

model = DNANet(
    num_classes=1,
    input_channels=3,
    block=Res_CBAM_block,
    num_blocks=[2, 2, 2, 2],
    nb_filter=[16, 32, 64, 128, 256],
    deep_supervision=True,
)

checkpoint = torch.load(
    "model/mIoU__DNANet_Final_Dataset_epoch.pth.tar", map_location=device
)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)
model.eval()

print("DNANet loaded succesfully")


# preprocessing


transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# load Image
img_path = "data/256/data 3/images/0256.png"
img = Image.open(img_path).convert("RGB")

# Resize to training size
img = img.resize((256, 256), Image.BILINEAR)

# apply transform
img = transforms(img)

# add batch dimensions
img = img.unsqueeze(0).to(device)


# Forward Pass
with torch.no_grad():
    outputs = model(img)


# deep supervision take last output
pred = outputs[-1]

# DEMO VISUALIZATION LOGIC
# convert logits -> binary mask
# same as (pred> 0)* 255
binary_mask = (pred > 0).float() * 255.0

# move to cpu  and numpy
binary_mask = binary_mask[0, 0].cpu().numpy().astype(np.uint8)


# save binary mask
save_path = "load_model_out/0256.png"
Image.fromarray(binary_mask).save(save_path)


# Show Exactly like demo

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(Image.open(img_path).resize((256, 256)), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Prediction (binary)")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Demo-style output saved:", save_path)
