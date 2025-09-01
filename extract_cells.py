# extract_cells.py
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
from skimage.measure import label, regionprops

from utils.models.ResU_net import ResUNet
from utils.models.U_net import UNet


# Dataset (images only)
class CellSegmentationDataset(Dataset):
    def __init__(self, filelist, image_dir):
        self.filenames = filelist
        self.image_dir = image_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = imread(os.path.join(self.image_dir, fname)) / 255.0
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0), fname


def load_filenames(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def build_model(name, in_ch, out_ch):
    if name.lower() == "resunet":
        return ResUNet(in_channels=in_ch, out_channels=out_ch)
    elif name.lower() == "unet":
        return UNet(in_channels=in_ch, out_channels=out_ch)
    else:
        raise ValueError(f"Unknown model {name}")


@torch.no_grad()
def run_crop_inference(model, loader, device, out_dir, min_size=20):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for x, fname in loader:
        x = x.to(device)
        pred = model(x)                       # (B, C, H, W)
        pred = torch.argmax(pred, dim=1)[0]   # (H, W)

        mask = pred.cpu().numpy().astype(np.uint8)

        # assume class 1 = interior
        cell_mask = (mask == 1).astype(np.uint8)

        # connected components
        labeled = label(cell_mask)
        props = regionprops(labeled)

        img = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)

        base = os.path.splitext(fname[0])[0]
        cell_id = 0
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            h, w = maxr - minr, maxc - minc
            if h < min_size or w < min_size:
                continue  # skip tiny specks

            crop = img[minr:maxr, minc:maxc]
            out_path = os.path.join(out_dir, f"{base}_cell{cell_id:03d}.png")
            imsave(out_path, crop)
            cell_id += 1

        print(f"{fname[0]} -> {cell_id} cells saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="norm_images")
    parser.add_argument("--test_list", type=str, default="test.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resunet", choices=["resunet", "unet"])
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="cropped_cells")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_size", type=int, default=20, help="minimum crop size (pixels)")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    filelist = load_filenames(args.test_list)
    dataset = CellSegmentationDataset(filelist, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)

    # model
    model = build_model(args.model, args.in_channels, args.out_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    run_crop_inference(model, loader, device, args.out_dir, min_size=args.min_size)


if __name__ == "__main__":
    main()
