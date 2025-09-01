# test_inference.py
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

from utils.models.ResU_net import ResUNet
from utils.models.U_net import UNet


# ----------------------------
# Dataset (same as training)
# ----------------------------
class CellSegmentationDataset(Dataset):
    def __init__(self, filelist, image_dir):
        self.filenames = filelist
        self.image_dir = image_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = imread(os.path.join(self.image_dir, filename)) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        return image, filename


def load_filenames(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_model(name: str, in_ch: int, out_ch: int):
    name = name.lower()
    if name in ["resunet", "res_u_net", "res-u-net"]:
        return ResUNet(in_channels=in_ch, out_channels=out_ch)
    elif name in ["unet", "u-net"]:
        return UNet(in_channels=in_ch, out_channels=out_ch)
    else:
        raise ValueError(f"Unknown model '{name}'")


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def run_inference(model, dataloader, device, out_dir, overlay=False):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    for x, fname in dataloader:
        x = x.to(device)
        pred = model(x)                   # (B, C, H, W)
        pred = torch.argmax(pred, dim=1)  # (B, H, W)

        mask = pred[0].cpu().numpy().astype(np.uint8)

        # save raw mask
        out_path = os.path.join(out_dir, os.path.splitext(fname[0])[0] + "_mask.png")
        imsave(out_path, mask)

        if overlay:
            img = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title("Input")
            plt.imshow(img, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(img, cmap="gray")
            plt.imshow(mask, alpha=0.5, cmap="jet")
            plt.axis("off")

            overlay_path = os.path.join(out_dir, os.path.splitext(fname[0])[0] + "_overlay.png")
            plt.savefig(overlay_path, bbox_inches="tight")
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="norm_images")
    parser.add_argument("--test_list", type=str, default="test.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resunet", choices=["resunet", "unet"])
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="test_predictions")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    filelist = load_filenames(args.test_list)
    test_dataset = CellSegmentationDataset(filelist, args.image_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    # model
    model = build_model(args.model, args.in_channels, args.out_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    # inference
    run_inference(model, test_loader, device, args.out_dir, overlay=args.overlay)


if __name__ == "__main__":
    main()
