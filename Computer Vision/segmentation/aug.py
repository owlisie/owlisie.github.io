import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import jaccard_score, f1_score
import albumentations as A
import time

# ============================
# Dataset with Augmentation
# ============================
class AugmentedSputumSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(256, 256), augmentations=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.augmentations = augmentations  # augmentation 설정

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.rsplit('.', 1)[0] + '_labels.txt')

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_resized = image.resize(self.image_size)
        image_np = np.array(image_resized)

        # 마스크 초기화
        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)  # (H, W)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                points = [tuple(map(float, line.strip().split(','))) for line in f if ',' in line]
                if len(points) >= 3:
                    sx = self.image_size[0] / original_size[0]
                    sy = self.image_size[1] / original_size[1]
                    scaled = [(int(x * sx), int(y * sy)) for x, y in points]
                    polygon = np.array(scaled, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [polygon], 1)

        # Augmentation 수행
        if self.augmentations:
            augmented = self.augmentations(image=image_np, mask=mask)
            image_resized = Image.fromarray(augmented['image'])
            mask = augmented['mask']

        # PyTorch tensor로 변환
        image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor
def augment(image, mask):
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
    return image, mask
    
class SputumSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(256, 256), augmentations=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.augmentations = augmentations
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.rsplit('.', 1)[0] + '_labels.txt')

        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = image.resize(self.image_size)
        image_np = np.array(image)

        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                points = [tuple(map(float, line.strip().split(','))) for line in f if ',' in line]
                if len(points) >= 3:
                    sx = self.image_size[0] / original_size[0]
                    sy = self.image_size[1] / original_size[1]
                    scaled = [(int(x*sx), int(y*sy)) for x, y in points]
                    polygon = np.array(scaled, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [polygon], 1)

        if self.augmentations:
            image_np, mask = augment(image_np, mask)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return image_tensor, mask_tensor

# ============================
# UNet Model
# ============================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = block(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.up(x2)
        x4 = self.dec1(x3)
        return torch.sigmoid(self.out(x4))

# ============================
# Train Function
# ============================
def train(model, dataset, device, epochs=10, lr=1e-3, batch_size=16, save_path="model.pth"):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = F.binary_cross_entropy(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {loss_sum / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ============================
# Evaluation & Visualization
# ============================
def evaluate_and_visualize(model, dataset, device, save_img_path="result_overlay.png"):
    model.eval()
    model.to(device)

    idx = np.random.randint(len(dataset))
    image_tensor, mask_tensor = dataset[idx]
    image = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)[0, 0].cpu().numpy()

    image_np = image_tensor.permute(1, 2, 0).numpy() * 255
    image_np = image_np.astype(np.uint8)
    pred_mask = (pred > 0.5).astype(np.uint8)
    true_mask = mask_tensor[0].numpy().astype(np.uint8)

    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    if np.sum(true_flat) > 0 or np.sum(pred_flat) > 0:
        iou = jaccard_score(true_flat, pred_flat, zero_division=0)
        dice = f1_score(true_flat, pred_flat, zero_division=0)
    else:
        iou = dice = 1.0

    print(f"Sample Evaluation: IoU = {iou:.4f}, Dice = {dice:.4f}")

    overlay = image_np.copy()
    overlay[true_mask == 1] = [255, 0, 0]      # Red: ground truth
    overlay[pred_mask == 1] = [0, 255, 0]      # Green: prediction

    cv2.imwrite(save_img_path, overlay[:, :, ::-1])  # RGB → BGR
    print(f"Overlay saved to {save_img_path}")


from sklearn.metrics import precision_score, recall_score

def evaluate_and_visualize_samples(model, dataset, device, save_dir="comparison_results_aug"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model.to(device)

    indices = np.arange(len(dataset))

    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        image = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image)[0, 0].cpu().numpy()

        image_np = image_tensor.permute(1, 2, 0).numpy() * 255
        image_np = image_np.astype(np.uint8)
        pred_mask = (pred > 0.5).astype(np.uint8)
        true_mask = mask_tensor[0].numpy().astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title("Original")
        axs[1].imshow(true_mask, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title("Prediction")

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_result_{i+1}.png"))
        plt.close()
    print(f"Saved 10 visual comparison results to '{save_dir}/'")

    ious, dices, precisions, recalls = [], [], [], []
    inference_times = []

    with torch.no_grad():
        for img, mask in test_loader:
            img = img.to(device)
            start_time = time.time()
            pred = model(img).cpu().numpy() > 0.5
            end_time = time.time()
            inference_times.append(end_time - start_time)

            true = mask.numpy()

            for p, t in zip(pred, true):
                p_flat = p[0].flatten()
                t_flat = t[0].flatten()

                if np.sum(t_flat) > 0 or np.sum(p_flat) > 0:
                    ious.append(jaccard_score(t_flat, p_flat, zero_division=0))
                    dices.append(f1_score(t_flat, p_flat, zero_division=0))
                    precisions.append(precision_score(t_flat, p_flat, zero_division=0))
                    recalls.append(recall_score(t_flat, p_flat, zero_division=0))
                else:
                    ious.append(1.0)
                    dices.append(1.0)
                    precisions.append(1.0)
                    recalls.append(1.0)

    return ious, dices, precisions, recalls, inference_times

# ============================
# Main
# ============================

import warnings

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    image_dir = "data/images"
    label_dir = "data/labels"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    image_size = (256, 256)

    test_image_dir = "data/test/images"
    test_label_dir = "data/test/labels"

    test_subset = Subset(SputumSegmentationDataset(test_image_dir, test_label_dir, image_size, augmentations=False), range(25))
    test_loader = DataLoader(test_subset, batch_size=1)

    # Augmentation 정의
    transform = A.Compose([
        A.RandomCrop(width=224, height=224, always_apply=True),   # 크기 변경 (Random crop)
        A.HorizontalFlip(p=0.5),                                  # 수평 반전
        A.RandomBrightnessContrast(p=0.2),                        # 밝기/대비 변화
        A.Rotate(limit=45, p=0.5),                                # 회전 (±45도)
        A.GaussianBlur(p=0.2),                                    # Gaussian blur
        A.CLAHE(p=0.2),                                           # Contrast Limited Adaptive Histogram Equalization
        A.RandomGamma(p=0.2),                                     # Gamma 보정
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),  # 이동/비율 변경/회전
    ])

    dataset = AugmentedSputumSegmentationDataset(image_dir, label_dir, image_size=(256, 256), augmentations=transform)
    model = UNet()

    train(model, dataset, device, epochs=100, save_path="./weight/unet_aug.pth")

    # 불러오기
    model.load_state_dict(torch.load("./weight/unet_aug.pth", map_location=device))
    test_dataset = SputumSegmentationDataset(test_image_dir, test_label_dir, image_size, augmentations=False)

    # 평가 및 시각화
    test_ious, test_dices, test_precisions, test_recalls, inference_times = evaluate_and_visualize_samples(model, test_dataset, device)

    print(f"Avg test IoU      : {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}")
    print(f"Avg test Dice     : {np.mean(test_dices):.4f} ± {np.std(test_dices):.4f}")
    print(f"Avg test Precision: {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")
    print(f"Avg test Recall   : {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")

    print(f"\n=== Inference Time ===")
    print(f"Avg time per image: {np.mean(inference_times)*1000:.2f} ms")
