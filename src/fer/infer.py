import os
from typing import List, Optional, cast

import torch
from PIL import Image
from torchvision import transforms

from .models import create_model


def load_checkpoint(ckpt_path: str, device: Optional[str] = None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"]
    model = create_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])  
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, class_names, device


def preprocess(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    out = tf(img)
    return cast(torch.Tensor, out)


def predict_images(ckpt_path: str, image_paths: List[str], img_size: int = 224):
    model, class_names, device = load_checkpoint(ckpt_path)
    results = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = preprocess(img, img_size)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            score, pred = prob.max(dim=1)
        results.append({
            "path": p,
            "label": class_names[pred.item()],
            "score": float(score.item()),
        })
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--images", type=str, nargs='+', required=True)
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()

    res = predict_images(args.ckpt, args.images, args.img_size)
    for r in res:
        print(f"{r['path']}: {r['label']} ({r['score']:.3f})")
