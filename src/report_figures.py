"""
Aggregate results from all three trained models into the tables and figures the
report needs. Run AFTER training all models.

Produces:
  outputs/metrics/comparison.csv        (mean metrics per model)
  outputs/metrics/per_label_AUROC.csv   (per-label AUROC, all models side by side)
  outputs/figures/learning_curves.png
  outputs/figures/auroc_comparison.png

Usage:
    python -m src.report_figures
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

MODELS = ["resnet18", "vgg19", "customcnn"]


def main():
    C.ensure_dirs()
    rows, per_label = [], {}
    histories = {}

    for m in MODELS:
        sj = C.METRIC_DIR / f"{m}_summary.json"
        pl = C.METRIC_DIR / f"{m}_per_label.csv"
        if not sj.exists():
            print(f"[skip] {m}: no summary yet (train it first)")
            continue
        summ = json.load(open(sj))
        rows.append({"model": m, **summ["test"]})
        histories[m] = summ["history"]
        if pl.exists():
            df = pd.read_csv(pl)
            per_label[m] = df.set_index("label")["AUROC"]

    if rows:
        comp = pd.DataFrame(rows).set_index("model")
        comp.to_csv(C.METRIC_DIR / "comparison.csv")
        print("\n=== Model comparison (test) ===")
        print(comp[["mean_AUROC", "mean_PR_AUC", "micro_F1", "macro_F1"]].round(4))

    if per_label:
        pl_df = pd.DataFrame(per_label)
        pl_df.to_csv(C.METRIC_DIR / "per_label_AUROC.csv")

    # Figures
    try:
        import matplotlib.pyplot as plt
        if histories:
            plt.figure(figsize=(7, 5))
            for m, h in histories.items():
                ep = [r["epoch"] for r in h]
                au = [r["mean_AUROC"] for r in h]
                plt.plot(ep, au, marker="o", label=m)
            plt.xlabel("epoch"); plt.ylabel("val mean AUROC")
            plt.title("Learning curves"); plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(C.FIGURE_DIR / "learning_curves.png", dpi=120)
            plt.close()
        if per_label:
            pl_df = pd.DataFrame(per_label)
            pl_df.plot(kind="barh", figsize=(8, 9))
            plt.xlabel("AUROC"); plt.title("Per-label AUROC by model")
            plt.tight_layout()
            plt.savefig(C.FIGURE_DIR / "auroc_comparison.png", dpi=120)
            plt.close()
        print(f"[figures] saved to {C.FIGURE_DIR}")
    except Exception as e:
        print(f"[figures] skipped ({e})")


if __name__ == "__main__":
    main()
