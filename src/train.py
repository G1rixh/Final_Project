"""
Training entry point.

Usage (run once per model):
    python -m src.train --model resnet18
    python -m src.train --model vgg19
    python -m src.train --model customcnn

Handles class imbalance via weighted BCEWithLogitsLoss (pos_weight) or focal
loss, tracks runs with MLflow if available, checkpoints best val mean-AUROC,
and saves learning-curve data for the report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
from src import data as D
from src import models as M
from src import evaluate as E


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def _try_mlflow():
    try:
        import mlflow
        return mlflow
    except Exception:
        return None


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        ps.append(prob); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(model_name: str):
    C.ensure_dirs()
    D.set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] model={model_name} device={device}")

    df = D.parse_metadata()
    train_df, val_df, test_df = D.make_splits(df)
    print(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    tr_loader, va_loader, te_loader = D.build_loaders(train_df, val_df, test_df)

    pw = D.pos_weights(train_df).to(device)
    model = M.get_model(model_name).to(device)
    criterion = (FocalLoss(pos_weight=pw) if C.USE_FOCAL_LOSS
                 else nn.BCEWithLogitsLoss(pos_weight=pw))
    optim = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=C.EPOCHS)

    mlflow = _try_mlflow()
    if mlflow:
        mlflow.start_run(run_name=model_name)
        mlflow.log_params(dict(model=model_name, lr=C.LR, epochs=C.EPOCHS,
                               batch_size=C.BATCH_SIZE, focal=C.USE_FOCAL_LOSS))

    history, best_auroc = [], -1.0
    ckpt_path = C.CHECKPOINT_DIR / f"{model_name}_best.pt"

    for epoch in range(1, C.EPOCHS + 1):
        model.train(); t0 = time.time(); running = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optim.step()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(tr_loader.dataset)

        y_true, y_prob = _collect(model, va_loader, device)
        val = E.summary_metrics(y_true, y_prob)
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(f"  epoch {epoch:02d} | loss {train_loss:.4f} | "
              f"val mAUROC {val['mean_AUROC']:.4f} | {time.time()-t0:.0f}s")
        if mlflow:
            mlflow.log_metrics({"train_loss": train_loss, **val}, step=epoch)

        if val["mean_AUROC"] > best_auroc:
            best_auroc = val["mean_AUROC"]
            torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                        "val_mean_auroc": best_auroc}, ckpt_path)

    # Final test evaluation with best checkpoint
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    y_true, y_prob = _collect(model, te_loader, device)
    test_summary = E.summary_metrics(y_true, y_prob)
    test_summary["ECE"] = E.expected_calibration_error(y_true, y_prob)
    per_label = E.per_label_table(y_true, y_prob)

    per_label.to_csv(C.METRIC_DIR / f"{model_name}_per_label.csv", index=False)
    json.dump({"history": history, "test": test_summary},
              open(C.METRIC_DIR / f"{model_name}_summary.json", "w"), indent=2)
    np.savez(C.METRIC_DIR / f"{model_name}_preds.npz", y_true=y_true, y_prob=y_prob)

    print(f"[test] {model_name} mean AUROC = {test_summary['mean_AUROC']:.4f}")
    if mlflow:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_summary.items()
                            if isinstance(v, (int, float))})
        mlflow.end_run()
    return test_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=list(M.MODEL_REGISTRY.keys()))
    args = ap.parse_args()
    train_one(args.model)
