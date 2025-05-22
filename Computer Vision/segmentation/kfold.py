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
from sklearn.model_selection import KFold

import albumentations as A
from sklearn.metrics import precision_score, recall_score

import time

# ============================
# Data Augmentation Utilities
# ============================
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

# ============================
# Dataset
# ============================
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
# Train One Fold
# ============================
def train_one_fold(model, train_loader, val_loader, device, epochs=100, lr=1e-3, save_path='./output/seg_unet.pth'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = F.binary_cross_entropy(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    ious, dices, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device)
            pred = model(img).cpu().numpy() > 0.5
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

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return np.mean(ious), np.mean(dices), np.mean(precisions), np.mean(recalls)


# ============================
# Visualize 10 Random Samples
# ============================
def evaluate_and_visualize_samples(model, dataset, device, save_dir="comparison_results_kfold_noseg"):
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


import warnings

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

# ============================
# Main (K-Fold)
# ============================

if __name__ == "__main__":
    image_dir = "data/images"
    label_dir = "data/labels"
    image_size = (256, 256)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    test_image_dir = "data/test/images"
    test_label_dir = "data/test/labels"

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

    # full_dataset = AugmentedSputumSegmentationDataset(image_dir, label_dir, image_size=(256, 256), augmentations=True)

    full_dataset = SputumSegmentationDataset(image_dir, label_dir, image_size, augmentations=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    all_ious, all_dices = [], []
    all_precisions, all_recalls = [], []

    test_subset = Subset(SputumSegmentationDataset(test_image_dir, test_label_dir, image_size, augmentations=False), range(25))
    test_loader = DataLoader(test_subset, batch_size=1)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n== Fold {fold+1} ==")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(SputumSegmentationDataset(image_dir, label_dir, image_size, augmentations=False), val_idx)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1)

        model = UNet()

        iou, dice, precision, recall = train_one_fold(model, train_loader, val_loader, device, epochs=100, save_path='weight/unet_kfold_noseg.pth')
        print(f"Fold {fold+1} → IoU: {iou:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        all_ious.append(iou)
        all_dices.append(dice)
        all_precisions.append(precision)
        all_recalls.append(recall)

    print(f"\n=== K-Fold Results ===")

    print(f"Avg IoU      : {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
    print(f"Avg Dice     : {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    print(f"Avg Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
    print(f"Avg Recall   : {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")


    test_dataset = SputumSegmentationDataset(test_image_dir, test_label_dir, image_size, augmentations=False)
    model.load_state_dict(torch.load("weight/unet_kfold_noseg.pth", map_location=device))

    test_ious, test_dices, test_precisions, test_recalls, inference_times = evaluate_and_visualize_samples(model, test_dataset, device)

    print(f"Avg test IoU      : {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}")
    print(f"Avg test Dice     : {np.mean(test_dices):.4f} ± {np.std(test_dices):.4f}")
    print(f"Avg test Precision: {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")
    print(f"Avg test Recall   : {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")

    print(f"\n=== Inference Time ===")
    print(f"Avg time per image: {np.mean(inference_times)*1000:.2f} ms")