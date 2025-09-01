# training.py
import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # (not used unless you extend to DDP)
import numpy as np
from skimage.io import imread
from tqdm import tqdm

# ---- your models / utils ----
from utils.models.U_net import UNet
# IMPORTANT: ensure this import matches your filename:
#   if file is utils/models/resunet.py -> use "resunet"
#   if file is utils/models/ResU_net.py -> keep "ResU_net"
from utils.models.ResU_net import ResUNet

from utils.crossentropy import weighted_crossentropy
from utils.metrics import channel_precision, channel_recall, categorical_accuracy


# ----------------------------
# Dataset
# ----------------------------
class CellSegmentationDataset(Dataset):
    def __init__(self, filelist, image_dir, label_dir):
        self.filenames = filelist
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = _resolve_file(self.image_dir, filename)
        lbl_path = _resolve_file(self.label_dir, filename)

        image = imread(img_path) / 255.0
        label = imread(lbl_path)  # (H, W, 3) or (H, W)

        if label.ndim == 2:
            one_hot = np.zeros((3, *label.shape), dtype=np.float32)
            for c in range(3):
                one_hot[c] = (label == c)
            label = one_hot
        else:
            # HWC -> CHW
            label = label.transpose(2, 0, 1) / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(label, dtype=torch.float32)               # (3, H, W)

        return image, label


# ----------------------------
# Helpers
# ----------------------------
# --- replace your load_filenames with this ---
def load_filenames(txt_path):
    out = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


# --- add these helpers into your training.py ---
import os

COMMON_IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def _resolve_file(root_dir, name):
    """
    Try to resolve a file path robustly:
    - if 'name' is an absolute or already has extension and exists, return it
    - else try root_dir/name
    - else if no extension, try common image extensions under root_dir
    """
    candidates = []

    # direct
    if os.path.isabs(name):
        candidates.append(name)
    else:
        candidates.append(os.path.join(root_dir, name))

    # if no extension, try common ones
    base, ext = os.path.splitext(name)
    if ext == "":
        for e in COMMON_IMG_EXTS:
            if os.path.isabs(name):
                candidates.append(base + e)
            else:
                candidates.append(os.path.join(root_dir, base + e))

    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            return p

    # If any candidate exists but is a directory, surface that clearly
    for p in candidates:
        if os.path.isdir(p):
            raise OSError(f"Path resolves to a directory, not a file: {p}")

    # Otherwise, not found
    raise FileNotFoundError(f"Could not find image file for '{name}' under '{root_dir}'. "
                            f"Tried: {candidates}")


def build_model(name: str, in_ch: int, out_ch: int) -> nn.Module:
    name = name.lower()
    if name in ["resunet", "res_u_net", "res-u-net"]:
        return ResUNet(in_channels=in_ch, out_channels=out_ch)
    elif name in ["unet", "u-net"]:
        return UNet(in_channels=in_ch, out_channels=out_ch)
    else:
        raise ValueError(f"Unknown model '{name}'")


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = weighted_crossentropy(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = weighted_crossentropy(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    accs = []
    prec_bg, prec_int, prec_bd = [], [], []
    rec_bg, rec_int, rec_bd = [], [], []

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)

        accs.append(float(categorical_accuracy(y, pred).item()))

        prec_bg.append(float(channel_precision(y, pred, 0).item()))
        rec_bg.append(float(channel_recall(y, pred, 0).item()))

        prec_int.append(float(channel_precision(y, pred, 1).item()))
        rec_int.append(float(channel_recall(y, pred, 1).item()))

        prec_bd.append(float(channel_precision(y, pred, 2).item()))
        rec_bd.append(float(channel_recall(y, pred, 2).item()))

    if len(accs) == 0:
        # empty dataloader
        return None

    results = {
        "accuracy": np.mean(accs).item(),
        "background_precision": np.mean(prec_bg).item(),
        "background_recall": np.mean(rec_bg).item(),
        "interior_precision": np.mean(prec_int).item(),
        "interior_recall": np.mean(rec_int).item(),
        "boundary_precision": np.mean(prec_bd).item(),
        "boundary_recall": np.mean(rec_bd).item(),
    }

    results["macro_precision"] = float(np.mean([
        results["background_precision"],
        results["interior_precision"],
        results["boundary_precision"]
    ]))
    results["macro_recall"] = float(np.mean([
        results["background_recall"],
        results["interior_recall"],
        results["boundary_recall"]
    ]))
    return results


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory=False):
    # persistent_workers works only if num_workers > 0
    persistent = True if num_workers and num_workers > 0 else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None
    )


def maybe_read_list(path):
    return load_filenames(path) if (path and os.path.isfile(path)) else None


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="norm_images")
    parser.add_argument("--label_dir", type=str, default="boundary_labels")

    parser.add_argument("--train_list", type=str, default="training.txt")
    parser.add_argument("--dev_list",   type=str, default="validation.txt")  # "dev"
    parser.add_argument("--test_list",  type=str, default="test.txt")        # "test"

    parser.add_argument("--model", type=str, default="resunet", choices=["resunet", "unet"])
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA only)")

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Data lists
    train_list = maybe_read_list(args.train_list)
    dev_list = maybe_read_list(args.dev_list)     # "validation" / "dev"
    test_list = maybe_read_list(args.test_list)

    if train_list is None:
        raise FileNotFoundError(f"Train list not found: {args.train_list}")

    # Datasets
    train_dataset = CellSegmentationDataset(train_list, args.image_dir, args.label_dir)
    dev_dataset   = CellSegmentationDataset(dev_list,   args.image_dir, args.label_dir) if dev_list else None
    test_dataset  = CellSegmentationDataset(test_list,  args.image_dir, args.label_dir) if test_list else None

    # DataLoaders (multiprocessing via num_workers)
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    dev_loader = make_loader(
        dev_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=args.num_workers
    ) if dev_dataset else None
    test_loader = make_loader(
        test_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=args.num_workers
    ) if test_dataset else None

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.in_channels, args.out_channels).to(device)

    # Optional multi-GPU (DataParallel keeps this simple)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Training loop
    best_dev_score = -np.inf
    metrics_log = []
    torch.manual_seed(42)
    torch.set_num_threads(args.num_workers)
    torch.set_num_interop_threads(args.num_workers)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device=device, scaler=scaler)

        # Evaluate on dev (validation)
        dev_results = evaluate(model, dev_loader, device=device) if dev_loader else None
        test_results = None
        #test_results = evaluate(model, test_loader, device=device) if test_loader else None

        # Print nicely
        print(f"Train loss: {train_loss:.4f}")
        if dev_results:
            print("Dev metrics:")
            print(f"  Acc: {dev_results['accuracy']:.4f} | "
                  f"Macro P/R: {dev_results['macro_precision']:.4f} / {dev_results['macro_recall']:.4f}")
        else:
            print("Dev metrics: [no dev set]")

        if test_results:
            print("Test metrics:")
            print(f"  Acc: {test_results['accuracy']:.4f} | "
                  f"Macro P/R: {test_results['macro_precision']:.4f} / {test_results['macro_recall']:.4f}")
        else:
            print("Test metrics: [no test set]")

        # Save latest checkpoint
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        latest_ckpt = os.path.join(args.out_dir, f"{args.model}_epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), latest_ckpt)

        # Save best on dev by macro F (here average of macro P/R as a surrogate if F not defined)
        if dev_results:
            score = 0.5 * (dev_results["macro_precision"] + dev_results["macro_recall"])
            if score > best_dev_score:
                best_dev_score = score
                best_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"✔ Saved best dev checkpoint to {best_path}")

        # Log metrics to JSONL
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev": dev_results,
            "test": test_results,
            "checkpoint": latest_ckpt,
            "timestamp": tag,
        }
        metrics_log.append(row)
        with open(os.path.join(args.out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    print("\nTraining complete.")
    if dev_loader:
        print(f"Best dev score (avg of macro P/R): {best_dev_score:.4f}")

    # test on best model

    if test_loader and dev_loader:
        print("\nEvaluating best dev model on test set...")
        best_model = build_model(args.model, args.in_channels, args.out_channels).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        best_model.eval()
        final_test_results = evaluate(best_model, test_loader, device=device)
        if final_test_results:
            print("Final test metrics:")
            print(f"  Acc: {final_test_results['accuracy']:.4f} | "
                  f"Macro P/R: {final_test_results['macro_precision']:.4f} / {final_test_results['macro_recall']:.4f}")
        else:
            print("Final test metrics: [no test set]")
if __name__ == "__main__":
    main()