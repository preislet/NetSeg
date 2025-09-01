# test_inference.py
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
import warnings

from utils.models.ResU_net import ResUNet
from utils.models.U_net import UNet
from utils.models.AttentionU_net import AttentionUNet

# silence low-contrast warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


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
    elif name in ["attentionunet", "attention_unet", "attention-u-net"]:
        return AttentionUNet(in_channels=in_ch, out_channels=out_ch)
    else:
        raise ValueError(f"Unknown model '{name}'")


# ----------------------------
# Color utilities
# ----------------------------
def mask_to_rgb(mask_hw: np.ndarray,
                interior_class: int = 1,
                boundary_class: int = 2,
                boundary_thickness: int = 1):
    """Convert class-index mask to RGB (red bg, green interior, cyan boundary)."""
    from skimage.morphology import binary_dilation, disk

    mask = mask_hw.astype(np.int32)
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # background = red
    rgb[..., 0] = 255

    # interior = green
    interior = (mask == interior_class)
    rgb[interior, :] = (0, 255, 0)

    # boundary = cyan
    boundary = (mask == boundary_class)
    if boundary_thickness > 1 and boundary.any():
        boundary = binary_dilation(boundary, disk(1), iterations=boundary_thickness - 1)
    rgb[boundary, :] = (0, 255, 255)

    return rgb


def stack_with_labels(panels, labels, header_h=28, pad=6,
                      bg=(0, 0, 0), fg=(255, 255, 255)):
    """
    Stack panels horizontally, adding a header band with text above each.
    panels: list of (H,W,3) uint8
    labels: list of strings with same length
    """
    from PIL import Image, ImageDraw, ImageFont

    assert len(panels) == len(labels) and len(panels) > 0
    H, W, _ = panels[0].shape
    for p in panels:
        assert p.shape == (H, W, 3), "All panels must have the same (H,W,3)"

    total_w = len(panels) * W + (len(panels) - 1) * pad
    total_h = header_h + H
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (240, 240, 240)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    x = 0
    for img, label in zip(panels, labels):
        canvas[0:header_h, x:x+W, :] = bg
        canvas[header_h:header_h+H, x:x+W, :] = img

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)

        # Pillow >=10 uses textbbox
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)

        tx = x + (W - text_w) // 2
        ty = (header_h - text_h) // 2
        draw.text((tx, ty), label, font=font, fill=fg)
        canvas = np.array(pil)

        x += W + pad

    return canvas


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def run_inference(model, dataloader, device, out_dir,
                  label_dir=None, overlay=False):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    for x, fname in dataloader:
        x = x.to(device)
        pred = model(x)                   # (B,C,H,W)
        pred = torch.argmax(pred, dim=1)  # (B,H,W)

        mask_pred = pred[0].cpu().numpy().astype(np.uint8)

        base = os.path.splitext(fname[0])[0]

        # save raw mask
        imsave(os.path.join(out_dir, base + "_mask.png"), mask_pred)

        if overlay:
            # input grayscale → RGB
            img_gray = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
            rgb_input = np.stack([img_gray]*3, axis=-1)

            # prediction overlay
            rgb_pred = mask_to_rgb(mask_pred)

            # ground truth overlay (if available)
            rgb_true = None
            if label_dir is not None:
                lbl_path = os.path.join(label_dir, fname[0])
                if os.path.isfile(lbl_path):
                    lbl_img = imread(lbl_path)
                    if lbl_img.ndim == 3:
                        # one-hot style PNG saved as 0/255 per channel
                        lbl_mask = np.argmax(lbl_img, axis=-1)
                    else:
                        lbl_mask = lbl_img.astype(np.uint8)
                    rgb_true = mask_to_rgb(lbl_mask)

            panels = [rgb_input, rgb_pred]
            labels = ["input", "prediction"]
            if rgb_true is not None:
                panels.append(rgb_true)
                labels.append("true")

            composite = stack_with_labels(
                panels, labels,
                header_h=28, pad=6,
                bg=(30, 30, 30),
                fg=(255, 255, 255)
            )
            imsave(os.path.join(out_dir, base + "_overlay.png"),
                   composite, check_contrast=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="norm_images")
    parser.add_argument("--label_dir", type=str, default="boundary_labels",
                        help="Directory with ground truth boundary labels")
    parser.add_argument("--test_list", type=str, default="test.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="resunet", choices=["resunet", "unet", "attentionunet"])
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="test_predictions")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filelist = load_filenames(args.test_list)
    dataset = CellSegmentationDataset(filelist, args.image_dir)
    pin = (device.type == "cuda")
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=pin)

    model = build_model(args.model, args.in_channels, args.out_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    run_inference(model, loader, device,
                  args.out_dir, label_dir=args.label_dir,
                  overlay=args.overlay)


if __name__ == "__main__":
    main()
