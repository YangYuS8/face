import os
import json
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from .data import build_dataloaders
from .models import create_model


@dataclass
class TrainConfig:
    data_root: str
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.2
    val_split: float = 0.1
    test_split: float = 0.0
    mix_precision: bool = True
    out_dir: str = "./outputs"
    patience: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_tb: bool = True
    tb_dir: Optional[str] = None
    log_wandb: bool = False
    project: str = "fer-baseline"
    run_name: Optional[str] = None


class EarlyStopper:
    def __init__(self, patience: int = 7, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, device, scaler: Optional[GradScaler], mix_precision: bool):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if mix_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def fit(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    tb_dir = cfg.tb_dir or os.path.join(cfg.out_dir, "tb")
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=tb_dir) if cfg.log_tb else None

    # Optional: Weights & Biases
    if cfg.log_wandb:
        try:
            import wandb  # type: ignore[import-not-found]
            wandb.init(project=cfg.project, name=cfg.run_name, config=asdict(cfg))
        except Exception as e:
            print(f"W&B init failed: {e}")
    train_loader, val_loader, test_loader, num_classes, class_names = build_dataloaders(
        cfg.data_root,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        augment=True,
    )

    model = create_model(num_classes=num_classes, backbone=cfg.backbone, pretrained=cfg.pretrained, dropout=cfg.dropout)
    model.to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.mix_precision)

    es = EarlyStopper(patience=cfg.patience, mode="max")

    best_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pt")

    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device, scaler, cfg.mix_precision)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step()

        report = classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0, output_dict=True)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "report": report,
        })

        print(f"Epoch {epoch}/{cfg.epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # Log scalars
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metrics/val_acc", val_acc, epoch)
        if cfg.log_wandb:
            try:
                import wandb  # type: ignore[import-not-found]
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
            except Exception:
                pass

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
            }, best_path)
            print(f"Saved best to {best_path} (val_acc={val_acc:.4f})")

        if es.step(float(val_acc)):
            print("Early stopping triggered.")
            break

    # Save history
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Optional final test
    test_result = None
    if test_loader is not None:
        ckpt = torch.load(best_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"]) 
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, cfg.device)
        test_report = classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0, output_dict=True)
        test_result = {"test_loss": test_loss, "test_acc": test_acc, "report": test_report}
        with open(os.path.join(cfg.out_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)
        print(f"Test - loss: {test_loss:.4f} acc: {test_acc:.4f}")

    # Close loggers
    if writer is not None:
        writer.flush()
        writer.close()
    if cfg.log_wandb:
        try:
            import wandb  # type: ignore[import-not-found]
            wandb.finish()
        except Exception:
            pass

    return best_path, history, test_result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Dataset root: train/val[/test] or flat class dirs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.0)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging")
    p.add_argument("--tb_dir", type=str, default=None, help="TensorBoard log dir (default outputs/tb)")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--project", type=str, default="fer-baseline")
    p.add_argument("--run_name", type=str, default=None)
    args = p.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
        val_split=args.val_split,
        test_split=args.test_split,
        mix_precision=not args.no_amp,
        out_dir=args.out_dir,
        log_tb=not args.no_tb,
        tb_dir=args.tb_dir,
        log_wandb=args.wandb,
        project=args.project,
        run_name=args.run_name,
    )

    fit(cfg)
