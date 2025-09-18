import json
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from .data import build_dataloaders
from .models import create_model


def evaluate_checkpoint(
    ckpt_path: str,
    data_root: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.0,
    test_split: float = 0.0,
    split: str = "auto",  # 'val' | 'test' | 'auto'
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"]

    model = create_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _, val_loader, test_loader, _, _ = build_dataloaders(
        data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        augment=False,
    )

    criterion = nn.CrossEntropyLoss()

    def _eval(loader, tta: bool = False):
        model.eval()
        loss_sum = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="eval", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                if tta:
                    outputs_flipped = model(torch.flip(images, dims=[3]))
                    outputs = (outputs + outputs_flipped) / 2.0
                loss = criterion(outputs, labels)
                loss_sum += loss.item() * images.size(0)
                pred = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
        return loss_sum / len(loader.dataset), y_true, y_pred

    # Choose split according to preference
    if split == "test" and test_loader is not None:
        loader = test_loader
    elif split == "val" and val_loader is not None:
        loader = val_loader
    else:
        loader = test_loader if test_loader is not None else val_loader

    if loader is None:
        raise RuntimeError(
            "No evaluation data found. Ensure your DATA_ROOT has 'val/' or 'test/' or provide --val_split/--test_split > 0 when using a single-folder dataset."
        )
    try:
        ds_len = len(loader.dataset)  # type: ignore[arg-type]
    except Exception:
        ds_len = 0
    if ds_len == 0:
        raise RuntimeError(
            "Empty evaluation dataset. Please verify your folders or split ratios."
        )

    loss, y_true, y_pred = _eval(loader, tta=args.tta)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0)

    print(f"Loss: {loss:.4f} Acc: {acc:.4f}")
    print(report)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.0, help="If your dataset has no val/ folder, set a >0 ratio to split from a single-folder root (seed=42)")
    p.add_argument("--test_split", type=float, default=0.0, help="If you want a held-out test from a single-folder root")
    p.add_argument("--split", type=str, default="auto", choices=["auto", "val", "test"], help="Which split to evaluate")
    p.add_argument("--tta", action="store_true", help="Enable simple TTA (horizontal flip)")
    args = p.parse_args()

    evaluate_checkpoint(
        args.ckpt,
        args.data_root,
        args.img_size,
        args.batch_size,
        args.num_workers,
        args.val_split,
        args.test_split,
        args.split,
    )
