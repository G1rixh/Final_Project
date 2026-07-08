"""
Grad-CAM explainability.

Produces class-activation heatmaps overlaid on the original X-ray for a chosen
label. Uses a lightweight self-contained Grad-CAM (no extra dependency needed,
though pytorch-grad-cam works too).

Usage:
    python -m src.gradcam --model resnet18 --image path/to.png --label Cardiomegaly
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from src import models as M


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out): self.acts = out.detach()
    def _bwd(self, _m, gin, gout): self.grads = gout[0].detach()

    def __call__(self, x, class_idx):
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)        # GAP of grads
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(logits)[0, class_idx].item()


def overlay_and_save(image_path, cam, out_path):
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE))
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


def run(model_name, image_path, label, ckpt=None):
    from torchvision import transforms
    from PIL import Image
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = M.get_model(model_name, pretrained=False).to(device)
    ckpt = ckpt or (C.CHECKPOINT_DIR / f"{model_name}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])

    tfm = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    x = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    x.requires_grad_(True)

    cam_engine = GradCAM(model, M.gradcam_target_layer(model, model_name))
    cam, prob = cam_engine(x, C.LABELS.index(label))
    out = C.FIGURE_DIR / f"gradcam_{model_name}_{label}_{Path(image_path).stem}.png"
    overlay_and_save(image_path, cam, out)
    print(f"[gradcam] {label} prob={prob:.3f} -> {out}")
    return out, prob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True, choices=C.LABELS)
    args = ap.parse_args()
    run(args.model, args.image, args.label)
