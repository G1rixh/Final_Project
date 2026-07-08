# NIH ChestX-ray14: Multi-label Thoracic Disease Classification with CNNs + RAG-grounded LLM Interpretation

**Author:** B Suriya Naraiyanan · **Program:** GUVI / HCL Data Analytics · **Date:** ____

> Assistive research tool. Not a medical device and not a diagnosis.

---

## 1. Problem & approach
Multi-label classification of 14 thoracic findings on NIH ChestX-ray14 using three
CNNs (ResNet-18, VGG-19, custom), with Grad-CAM explainability and a RAG layer
that turns predictions into a structured, citation-backed, uncertainty-aware
interpretation. Findings are multi-hot; outputs are 14 independent sigmoids.

## 2. Data & leakage-safe splits
- Source: Kaggle NIH Chest X-ray Dataset (~112k frontal radiographs, ~30.8k patients).
- Labels text-mined from reports → noisy; severe class imbalance.
- Splits: official `train_val_list.txt` / `test_list.txt` (patient-disjoint), with a
  patient-grouped 10% validation carve-out. Leakage assertion enforced in code.
- Split sizes: train ____ / val ____ / test ____.
- Per-label positive counts: _(paste from training log)_

## 3. Preprocessing
Resize 224×224, ImageNet normalization, grayscale→3ch. Augmentation: mild affine
(±7°, 5% translate) + light brightness/contrast. **No horizontal flips** (anatomy).

## 4. Models & training
| Model | Pretrained | Loss | Epochs | LR |
|---|---|---|---|---|
| ResNet-18 | ImageNet | weighted BCE (pos_weight) | 8 | 1e-4 |
| VGG-19 | ImageNet | weighted BCE (pos_weight) | 8 | 1e-4 |
| Custom CNN | no | weighted BCE (pos_weight) | 8 | 1e-4 |

Optimizer AdamW + cosine schedule; best checkpoint by val mean AUROC; MLflow tracking.

## 5. Results — model comparison (test set)
_(paste `outputs/metrics/comparison.csv`)_

| Model | mean AUROC | mean PR-AUC | micro F1 | macro F1 | ECE |
|---|---|---|---|---|---|
| ResNet-18 | ____ | ____ | ____ | ____ | ____ |
| VGG-19 | ____ | ____ | ____ | ____ | ____ |
| Custom CNN | ____ | ____ | ____ | ____ | ____ |

**Primary metric (mean AUROC across 14 labels):** best = ____ (____).

## 6. Per-label AUROC / PR-AUC
_(paste `outputs/metrics/per_label_AUROC.csv` / `<model>_per_label.csv`)_

## 7. Learning curves & failure analysis
- Figure: `outputs/figures/learning_curves.png`
- Figure: `outputs/figures/auroc_comparison.png`
- Failure notes: worst labels (e.g. Hernia/Pneumonia — few positives, high variance),
  common confusions (Infiltration ↔ Consolidation ↔ Edema), and a couple of
  mispredicted examples with brief commentary.

## 8. Grad-CAM examples
Insert 3–5 overlays from `outputs/figures/`. For each: image, predicted label +
probability, and one line on whether the heatmap localizes a plausible region.

## 9. RAG-grounded LLM summaries (with citations)
Paste 2–3 generated summaries (`python -m src.rag.interpret`). Each must:
cite `[KB-...]` snippets, use uncertainty bands, avoid definitive diagnosis, and
end with the disclaimer. Note retrieval spot-check + citation-correctness check.

## 10. Limitations & safety
Label noise, single frontal view only, no clinical context, imbalance-driven
variance on rare labels, Grad-CAM is attribution not proof. The system is
assistive; every output carries the non-diagnostic disclaimer.

## 11. Reproducibility
Fixed seed (42), pinned `requirements.txt`, patient-wise splits, MLflow logs,
checkpoints saved per model.

## 12. Links
- GitHub: github.com/G1rixh/____
- (Optional) FastAPI demo: `uvicorn src.api:app`
