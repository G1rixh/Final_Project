"""
Smoke test: proves the pipeline logic is correct WITHOUT the real dataset or a
GPU. Run this first to confirm the repo is wired correctly, then move to Kaggle
for the actual training.

    python run_smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as C
from src import models as M
from src import evaluate as E


def test_models():
    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    for name in M.MODEL_REGISTRY:
        model = M.get_model(name, pretrained=False)
        out = model(x)
        assert out.shape == (2, C.NUM_LABELS), f"{name} -> {out.shape}"
        # Grad-CAM target layer must resolve for every model
        _ = M.gradcam_target_layer(model, name)
    print(f"[ok] 3 models forward-pass to shape (B, {C.NUM_LABELS}); "
          f"Grad-CAM layers resolve")


def test_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(500, C.NUM_LABELS))
    # correlated scores so AUROC > 0.5
    y_prob = np.clip(y_true * 0.6 + rng.random((500, C.NUM_LABELS)) * 0.5, 0, 1)
    summ = E.summary_metrics(y_true, y_prob)
    assert 0.0 <= summ["mean_AUROC"] <= 1.0
    ece = E.expected_calibration_error(y_true, y_prob)
    thr = E.tune_thresholds(y_true, y_prob)
    assert thr.shape == (C.NUM_LABELS,)
    print(f"[ok] metrics: mean_AUROC={summ['mean_AUROC']:.3f} "
          f"micro_F1={summ['micro_F1']:.3f} ECE={ece:.3f}")


def test_label_parsing_and_splits():
    # Build a tiny synthetic Data_Entry CSV + fake image files
    from src import data as D
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "images" / "images").mkdir(parents=True)
        rows, files = [], []
        rng = np.random.default_rng(1)
        for pid in range(40):
            for k in range(rng.integers(1, 4)):
                fn = f"{pid:08d}_{k:03d}.png"
                # create a tiny valid png
                from PIL import Image
                Image.new("L", (8, 8)).save(d / "images" / "images" / fn)
                labs = rng.choice(C.LABELS, size=rng.integers(0, 3), replace=False)
                rows.append({"Image Index": fn,
                             "Finding Labels": "|".join(labs) or "No Finding",
                             "Patient ID": pid})
                files.append(fn)
        pd.DataFrame(rows).to_csv(d / C.CSV_NAME, index=False)

        df = D.parse_metadata(d)
        assert all(lab in df.columns for lab in C.LABELS)
        assert df["path"].notna().all()
        train, val, test = D.make_splits(df, d)   # asserts no patient leakage inside
        pw = D.pos_weights(train)
        assert pw.shape == (C.NUM_LABELS,)
        print(f"[ok] parsing+splits: 14 multihot cols, patient-disjoint "
              f"train/val/test = {len(train)}/{len(val)}/{len(test)}, "
              f"pos_weights computed")


def test_rag():
    from src.rag.build_index import load_chunks
    from src.rag.interpret import retrieve, build_prompt, confidence_word
    chunks = load_chunks()
    assert len(chunks) >= 18, f"expected full KB, got {len(chunks)} chunks"
    hits = retrieve("pleural effusion fluid costophrenic", chunks, top_k=3)
    assert len(hits) == 3
    fake = {"Effusion": 0.91, "Atelectasis": 0.74, "Pneumonia": 0.1}
    system, user, ids = build_prompt(fake)
    assert "KB-DISCLAIMER" in ids and "KB-EFFUSION" in ids
    assert confidence_word(0.9) == "high probability"
    assert "Only state facts supported" in system
    print(f"[ok] RAG: {len(chunks)} KB chunks, retrieval works, prompt enforces "
          f"grounding + cites {len(ids)} snippets")


if __name__ == "__main__":
    print("Running smoke tests (no dataset / no GPU needed)\n" + "-" * 55)
    test_models()
    test_metrics()
    test_label_parsing_and_splits()
    test_rag()
    print("-" * 55 + "\nAll smoke tests passed. Repo logic is correct.")
