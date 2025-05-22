import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score

# ========== Dataset ==========
class SputumSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.image_size = image_size  # (width, height)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        # Load image and get original size
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        resized_image = image.resize(self.image_size)
        image_np = np.array(resized_image)

        # Resize label points proportionally
        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)  # shape: (height, width)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                points = [tuple(map(float, line.strip().split(','))) for line in f if ',' in line]
                if len(points) >= 3:
                    # Compute scale
                    scale_x = self.image_size[0] / original_size[0]
                    scale_y = self.image_size[1] / original_size[1]
                    scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]
                    scaled_points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [scaled_points], 1)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor


# ========== Model ==========
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU()
            )
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = conv_block(128, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        d1 = self.dec1(d1)
        out = torch.sigmoid(self.final(d1))
        return out

# ========== Training ==========
def train_model(model, dataset, device, epochs=20, batch_size=4, lr=1e-3, save_path='model.pth'):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = F.binary_cross_entropy(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ========== Evaluation ==========
def evaluate_model(model, dataset, device, threshold=0.5):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    ious, dices = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds = model(images).cpu().numpy() > threshold
            masks = masks.numpy()

            pred_flat = preds.flatten()
            true_flat = masks.flatten()

            if np.sum(true_flat) == 0 and np.sum(pred_flat) == 0:
                continue

            iou = jaccard_score(true_flat, pred_flat)
            dice = f1_score(true_flat, pred_flat)
            ious.append(iou)
            dices.append(dice)

    print(f"Mean IoU: {np.mean(ious):.4f}, Mean Dice: {np.mean(dices):.4f}")

# ========== Inference & Visualization ==========
def visualize_prediction(model, image_path, label_path, image_size=(256, 256), device='cuda'):
    image = Image.open(image_path).convert('RGB').resize(image_size)
    img_np = np.array(image)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    mask = np.zeros(image_size, dtype=np.uint8)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            points = [tuple(map(float, line.strip().split(','))) for line in f if ',' in line]
            if len(points) >= 3:
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], 1)

    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(pred > 0.5, cmap='gray')
    plt.title('Predicted Mask')

    plt.show()

# ========== Example Usage ==========
if __name__ == "__main__":
    image_dir = "data/images"
    label_dir = "data/labels"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SputumSegmentationDataset(image_dir, label_dir)
    model = UNet()

    # 학습
    train_model(model, dataset, device)

    # 평가
    evaluate_model(model, dataset, device)

    # 예측 시각화
    visualize_prediction(model, 
                         image_path=os.path.join(image_dir, "001_2차_220923.jpg"),
                         label_path=os.path.join(label_dir, "001_2차_220923_labels.txt"))
