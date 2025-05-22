import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from shapely.geometry import Polygon

class SputumSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, label_format='custom'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_format = label_format  # 'custom', 'coco', or 'yolo'
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_names)

    def _load_polygon(self, path, width, height):
        if not os.path.exists(path):
            return None  # No label

        with open(path, 'r') as f:
            lines = f.readlines()

        try:
            if self.label_format == 'custom':
                points = [tuple(map(float, l.strip().split(','))) for l in lines if ',' in l]
            elif self.label_format == 'coco':
                coords = list(map(float, lines[0].strip().split(',')))
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            elif self.label_format == 'yolo':
                # Format: class cx cy w h (normalized)
                class_id, cx, cy, w, h = map(float, lines[0].strip().split())
                cx *= width
                cy *= height
                w *= width
                h *= height
                points = [
                    (cx - w / 2, cy - h / 2),
                    (cx + w / 2, cy - h / 2),
                    (cx + w / 2, cy + h / 2),
                    (cx - w / 2, cy + h / 2),
                ]
            else:
                raise ValueError("Unsupported label format")

            # Check polygon validity
            if len(points) < 3:
                return None
            poly = Polygon(points)
            if not poly.is_valid or poly.area < 1:
                return None
            return np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        except Exception as e:
            print(f"[Warning] Skipping invalid polygon in {path}: {e}")
            return None

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        # Load image (BGR to RGB)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = self._load_polygon(label_path, width, height)
        if polygon is not None:
            cv2.fillPoly(mask, [polygon], 1)

        # Tensor conversion
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor
