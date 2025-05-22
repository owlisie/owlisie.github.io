import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def save_resized_mask_overlay_images(image_dir, label_dir, save_dir, image_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)
    image_names = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_name in tqdm(image_names):
        image_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '_labels.txt')

        # 원본 이미지 읽기
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_resized = image.resize(image_size)
        image_np = np.array(image_resized)

        # 마스크 생성
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                points = [tuple(map(float, line.strip().split(','))) for line in f if ',' in line]
                if len(points) >= 3:
                    sx = image_size[0] / original_size[0]
                    sy = image_size[1] / original_size[1]
                    scaled = [(int(x*sx), int(y*sy)) for x, y in points]
                    polygon = np.array(scaled, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [polygon], 1)

        # Overlay 마스크
        overlay = image_np.copy()
        overlay[mask == 1] = [255, 0, 0]  # Red for mask

        # 저장
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, overlay[:, :, ::-1])  # RGB → BGR

    print(f"Overlay images saved to: {save_dir}")


save_resized_mask_overlay_images(
    image_dir="data/images",
    label_dir="data/labels",
    save_dir="data/overlay_preview_resize",
    image_size=(256, 256)
)