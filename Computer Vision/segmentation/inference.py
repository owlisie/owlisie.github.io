import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import SputumSegmentationDataset
from train import train
from unet import UNet

def visualize_prediction(model, dataset, index, device):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[index]
        pred = model(image.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        image_np = image.permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(mask.squeeze().numpy(), cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred > 0.5, cmap='gray')
    axs[2].set_title("Prediction")
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SputumSegmentationDataset('./data/images/', './data/labels/', label_format='custom')  # 또는 'coco', 'yolo'
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

train(model, loader, optimizer, criterion, device, epochs=20)
visualize_prediction(model, dataset, index=0, device=device)