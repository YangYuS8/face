import os
from typing import Tuple, Optional, Dict, Sequence

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DEFAULT_MEAN_STD = {
    224: {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
}


def _build_transforms(img_size: int = 224, augment: bool = True):
    mean = DEFAULT_MEAN_STD.get(img_size, DEFAULT_MEAN_STD[224])["mean"]
    std = DEFAULT_MEAN_STD.get(img_size, DEFAULT_MEAN_STD[224])["std"]

    train_tf = [
        transforms.Resize((img_size, img_size)),
    ]
    if augment:
        train_tf = [
            transforms.Resize(int(img_size * 1.1)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    train_tf += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(train_tf), val_tf


def _infer_structure(data_root: str) -> Dict[str, Optional[str]]:
    """
    Detect dataset folder structure.
    Preferred:
        data_root/train, data_root/val, [data_root/test]
    Fallback:
        data_root/ contains class folders -> we will split into train/val
    """
    paths: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
    for split in ["train", "val", "test"]:
        p = os.path.join(data_root, split)
        if os.path.isdir(p):
            paths[split] = p
    if paths["train"] is None:
        # Try flat structure where data_root has class dirs
        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        # Heuristic: at least 2 subdirs = class folders
        if len(subdirs) >= 2:
            paths["train"] = data_root
    return paths


def build_dataloaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.0,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int, Tuple[str, ...]]:
    """
    Returns train_loader, val_loader, test_loader (optional), num_classes, class_names
    Accepted structures:
      - data_root/train, data_root/val, [data_root/test]
      - data_root/<class>/* (auto split train/val[/test])
    """
    assert os.path.isdir(data_root), f"data_root not found: {data_root}"
    train_tf, val_tf = _build_transforms(img_size, augment)

    paths = _infer_structure(data_root)

    if paths["train"] and paths["val"]:
        train_ds = datasets.ImageFolder(paths["train"], transform=train_tf)
        val_ds = datasets.ImageFolder(paths["val"], transform=val_tf)
        test_ds = datasets.ImageFolder(paths["test"], transform=val_tf) if paths["test"] else None
    else:
        # Single folder -> split on indices; use distinct datasets per split to apply different transforms
        root = paths["train"] or data_root
        base_eval_ds = datasets.ImageFolder(root, transform=val_tf)
        n_total = len(base_eval_ds)
        n_test = int(n_total * test_split) if test_split > 0 else 0
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test

        # Stable split
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_idx: Sequence[int] = perm[:n_train]
        val_idx: Sequence[int] = perm[n_train:n_train + n_val]
        test_idx: Sequence[int] = perm[n_train + n_val:n_train + n_val + n_test]

        train_base_ds = datasets.ImageFolder(root, transform=train_tf)
        val_base_ds = base_eval_ds  # already val_tf
        test_base_ds = datasets.ImageFolder(root, transform=val_tf) if n_test > 0 else None

        from torch.utils.data import Subset
        train_ds = Subset(train_base_ds, train_idx)
        val_ds = Subset(val_base_ds, val_idx)
        test_ds = Subset(test_base_ds, test_idx) if (test_base_ds is not None and n_test > 0) else None

    # Discover class names from a known ImageFolder
    if isinstance(train_ds, datasets.ImageFolder):
        class_names = tuple(train_ds.classes)
    else:
        # train_ds is likely Subset
        base = train_ds.dataset  # type: ignore[attr-defined]
        class_names = tuple(getattr(base, 'classes'))  # type: ignore[arg-type]
    num_classes = len(class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes, class_names
