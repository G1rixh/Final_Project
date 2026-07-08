# NIH ChestX-ray14 — Multi-label Classification + RAG-grounded LLM Interpretation

Multi-label thoracic disease classifier (14 findings) using three CNNs
(ResNet-18, VGG-19, custom CNN), with Grad-CAM explainability and a
Retrieval-Augmented Generation layer that produces structured, citation-backed,
uncertainty-aware interpretations of the model outputs — with a mandatory
non-diagnostic disclaimer.

> **Assistive research tool only. Not a medical device. Not a diagnosis.**

---

## Why training runs on Kaggle (not locally)

## Live notebook & model checkpoints
Full reproducible run (training, Grad-CAM, RAG output) + all three checkpoints:
<https://www.kaggle.com/code/g1rixh/final-project>

The dataset is ~112k images / ~42 GB and training three CNNs needs a GPU. The
**dataset is already mounted on Kaggle** ("NIH Chest X-rays"), with a free
P100/T4 — so the fastest path is: upload this repo to a Kaggle notebook, attach
the dataset, run. No 42 GB download.

## Repo layout

```
config.py                 # all paths, labels, hyperparameters, seed
run_smoke_test.py         # validates logic with no dataset/GPU (run first)
requirements.txt
src/
  data.py                 # CSV parse -> 14 multi-hot, PATIENT-WISE splits, loaders
  models.py               # resnet18 / vgg19 / customcnn factory (14 sigmoid out)
  train.py                # weighted-BCE/focal training, MLflow, checkpoints
  evaluate.py             # AUROC, PR-AUC, micro/macro F1, precision/recall, ECE
  gradcam.py              # Grad-CAM heatmap overlays
  report_figures.py       # comparison table + learning curves + AUROC chart
  api.py                  # optional FastAPI /predict demo
  rag/
    knowledge_base.md     # curated, citable medical notes (the only LLM source)
    build_index.py        # chunk -> embed -> FAISS index
    interpret.py          # retrieve + LLM (Gemini/Claude) grounded summary
report/
  report_template.md      # fill with YOUR real numbers + figures
```

## Quick start

**0. Confirm the logic (anywhere, ~10s, no GPU/data):**
```bash
pip install -r requirements.txt
python run_smoke_test.py
```

**1. Point at the data.** Edit `DATA_DIR` in `config.py`
(`/kaggle/input/data` on Kaggle, or your local folder).

**2. Train all three models** (run on the GPU box / Kaggle):
```bash
python -m src.train --model resnet18
python -m src.train --model vgg19
python -m src.train --model customcnn
```
Each saves a best checkpoint (`outputs/checkpoints/<model>_best.pt`), per-label
metrics, learning-curve history, and raw test predictions.

**3. Build the comparison tables + figures:**
```bash
python -m src.report_figures
```

**4. Grad-CAM on a few cases:**
```bash
python -m src.gradcam --model resnet18 --image /path/to/xray.png --label Cardiomegaly
```

**5. RAG index + grounded summaries:**
```bash
python -m src.rag.build_index
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY=... and set LLM_PROVIDER
python -m src.rag.interpret --demo
```

**6. (Optional) FastAPI demo:**
```bash
uvicorn src.api:app --reload      # then open /docs and POST an image
```

## Switching the LLM provider

In `config.py` set `LLM_PROVIDER = "gemini"` or `"anthropic"`. Provide the
matching key via `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`. Both are already wired.

## What an evaluator will check (and where it's handled)

| Criterion | Where |
|---|---|
| No patient leakage across splits | `data.make_splits` (official lists + grouped val + assertion) |
| Class imbalance handled | `data.pos_weights` + weighted BCE / focal in `train.py` |
| Primary metric = mean AUROC | `evaluate.summary_metrics` |
| PR-AUC, micro/macro F1, precision/recall | `evaluate.per_label_table` |
| Grad-CAM explainability | `gradcam.py` |
| RAG grounding (claims cite retrieved snippets) | `rag/interpret.build_prompt` |
| No definitive-diagnosis language + disclaimer | enforced in prompt + safety net in `interpret` |
| Reproducibility | fixed `SEED`, pinned `requirements.txt`, MLflow logging |

## Expected results (set expectations)

With transfer learning on the full set, mean AUROC typically lands ~**0.75–0.82**
(the original NIH baselines averaged ~0.75). The custom CNN will trail the
pretrained backbones — report that gap as a finding, not a bug. Rare labels
(Hernia, Pneumonia) show high variance; lead with PR-AUC there, not just AUROC.
