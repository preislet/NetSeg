# training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.io import imread
from torchvision import transforms
from tqdm import tqdm
from utils.models.U_net import UNet
from utils.models.ResU_net import ResUNet
from utils.crossentropy import weighted_crossentropy
from utils.metrics import channel_precision, channel_recall, categorical_accuracy


# Dataset
class CellSegmentationDataset(Dataset):
    def __init__(self, filelist, image_dir, label_dir):
        self.filenames = filelist
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = imread(os.path.join(self.image_dir, filename)) / 255.0
        label = imread(os.path.join(self.label_dir, filename))  # (H, W, 3)

        if label.ndim == 2:
            one_hot = np.zeros((3, *label.shape), dtype=np.float32)
            for c in range(3):
                one_hot[c] = (label == c)
            label = one_hot
        else:
            label = label.transpose(2, 0, 1) / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def load_filenames(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = weighted_crossentropy(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()

    accs = []
    prec_bg, prec_int, prec_bd = [], [], []
    rec_bg, rec_int, rec_bd = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            accs.append(categorical_accuracy(y, pred).item())

            prec_bg.append(channel_precision(y, pred, 0).item())
            rec_bg.append(channel_recall(y, pred, 0).item())

            prec_int.append(channel_precision(y, pred, 1).item())
            rec_int.append(channel_recall(y, pred, 1).item())

            prec_bd.append(channel_precision(y, pred, 2).item())
            rec_bd.append(channel_recall(y, pred, 2).item())

    results = {
        "accuracy": np.mean(accs),
        "background_precision": np.mean(prec_bg),
        "background_recall": np.mean(rec_bg),
        "interior_precision": np.mean(prec_int),
        "interior_recall": np.mean(rec_int),
        "boundary_precision": np.mean(prec_bd),
        "boundary_recall": np.mean(rec_bd),
    }

    results["macro_precision"] = np.mean([
        results["background_precision"],
        results["interior_precision"],
        results["boundary_precision"]
    ])
    results["macro_recall"] = np.mean([
        results["background_recall"],
        results["interior_recall"],
        results["boundary_recall"]
    ])
    return results


def main():
    image_dir = "norm_images"
    label_dir = "boundary_labels"

    train_list = load_filenames("training.txt")
    val_list = load_filenames("validation.txt")

    train_dataset = CellSegmentationDataset(train_list, image_dir, label_dir)
    val_dataset = CellSegmentationDataset(val_list, image_dir, label_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet(in_channels=1, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 2):
        print(f"\nEpoch {epoch}")
        loss = train_one_epoch(model, train_loader, optimizer, device=device)
        results = evaluate(model, val_loader, device=device)

        print("\nEvaluation metrics:")
        print(f"Accuracy:             {results['accuracy']:.4f}")
        print(f"Background  - P/R:    {results['background_precision']:.4f} / {results['background_recall']:.4f}")
        print(f"Interior    - P/R:    {results['interior_precision']:.4f} / {results['interior_recall']:.4f}")
        print(f"Boundary    - P/R:    {results['boundary_precision']:.4f} / {results['boundary_recall']:.4f}")
        print(f"Macro       - P/R:    {results['macro_precision']:.4f} / {results['macro_recall']:.4f}")

        torch.save(model.state_dict(), f"checkpoints/unet_epoch{epoch:02d}.pt")


if __name__ == "__main__":
    main()
