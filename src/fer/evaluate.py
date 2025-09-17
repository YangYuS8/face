import json
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from .data import build_dataloaders
from .models import create_model


def evaluate_checkpoint(ckpt_path: str, data_root: str, img_size: int = 224, batch_size: int = 64, num_workers: int = 4):
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
        val_split=0.0,  # assume val exists or use test
        test_split=0.0,
        augment=False,
    )

    criterion = nn.CrossEntropyLoss()

    def _eval(loader):
        model.eval()
        loss_sum = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="eval", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item() * images.size(0)
                pred = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
        return loss_sum / len(loader.dataset), y_true, y_pred

    if test_loader is not None:
        loader = test_loader
    else:
        loader = val_loader

    loss, y_true, y_pred = _eval(loader)
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
    args = p.parse_args()

    evaluate_checkpoint(args.ckpt, args.data_root, args.img_size, args.batch_size, args.num_workers)
