# NIH ChestX-ray14 — Curated Knowledge Base for RAG Grounding

> This file is the *only* source the LLM interpretation layer is allowed to cite.
> Each entry is a self-contained chunk. Citation markers in generated summaries
> refer to the `[ID]` shown in each heading.

---

## [KB-DISCLAIMER] Non-diagnostic disclaimer
This system is an automated, assistive research tool. It does NOT provide a
medical diagnosis and has not been validated or approved for clinical use. All
outputs are probabilistic and may be wrong. Final interpretation must be made by
a qualified radiologist or physician using the original images and full clinical
context. This summary must always be read together with this disclaimer.

## [KB-DATASET] About the ChestX-ray14 labels
The 14 labels were text-mined from free-text radiology reports using NLP, not
verified pixel-by-pixel by radiologists for every image. As a result the labels
contain noise and uncertainty. Several findings are rare, producing severe class
imbalance. Reported probabilities reflect the model's learned patterns on this
imperfect label set, not ground-truth clinical certainty.

## [KB-UNCERTAINTY] Reading the probabilities
Each finding carries a sigmoid probability in [0, 1]. Use uncertainty-aware
language: high probability (>= 0.80), moderate (0.50–0.79), low (< 0.50). A high
probability is a prompt for human review, never a confirmed diagnosis. Multiple
findings can co-occur; absence of a high score does not rule a condition out.

## [KB-LIMITS] Known model limitations
The model sees only a single resized frontal radiograph. It has no access to
lateral views, prior studies, clinical history, or labs. It can be misled by
imaging artifacts, rotation, projection (AP vs PA), patient positioning, support
devices, and image quality. Grad-CAM shows where the model looked, which may or
may not correspond to true pathology.

---

## [KB-ATELECTASIS] Atelectasis
Collapse or incomplete expansion of lung tissue. Typical radiographic signs:
volume loss, increased opacity, displacement of fissures, and shift of the
mediastinum or diaphragm toward the affected side. Often subtle and easily
confused with other opacities. Common and frequently co-occurs with effusion.

## [KB-CARDIOMEGALY] Cardiomegaly
Enlarged cardiac silhouette, classically a cardiothoracic ratio > 0.5 on a
PA view. AP projection and poor inspiration can falsely enlarge the heart
shadow, so projection matters. Suggests but does not confirm cardiac disease.

## [KB-EFFUSION] Effusion (pleural)
Fluid in the pleural space. Signs: blunting of the costophrenic angle, meniscus
sign, and homogeneous lower-zone opacity that may layer with positioning. Can
obscure underlying lung and co-occurs with atelectasis and consolidation.

## [KB-INFILTRATION] Infiltration
A broad, non-specific term for ill-defined increased opacity in the lung. Highly
heterogeneous and noisy in this dataset; overlaps with consolidation, edema, and
atelectasis. Low specificity; interpret cautiously.

## [KB-MASS] Mass
A discrete opacity >= 3 cm. Requires correlation with prior imaging and often CT
for characterization. Can be mimicked by overlapping structures or nipple
shadows. A flagged mass warrants radiologist review, not a benign/malignant call.

## [KB-NODULE] Nodule
A rounded opacity < 3 cm. Small nodules are easy to miss or to confuse with
vessels, bony structures, or skin lesions. Management depends on size and risk
factors and is outside this model's scope.

## [KB-PNEUMONIA] Pneumonia
Airspace opacity reflecting infection; radiographically overlaps heavily with
consolidation, atelectasis, and edema. Radiographic diagnosis of pneumonia is
clinical-radiologic and cannot be made from the image alone. Rare in this
dataset and noisy.

## [KB-PNEUMOTHORAX] Pneumothorax
Air in the pleural space. Signs: a visible visceral pleural line with absent lung
markings peripherally; a large/tension pneumothorax is a medical emergency.
Subtle apical pneumothoraces are easily missed; expiratory or upright views help.

## [KB-CONSOLIDATION] Consolidation
Alveolar filling producing dense opacity, often with air bronchograms. Overlaps
with pneumonia, edema, and hemorrhage; the pattern alone does not specify cause.

## [KB-EDEMA] Edema (pulmonary)
Fluid accumulation in the lung, commonly cardiogenic. Signs: bilateral perihilar
("bat-wing") opacities, Kerley B lines, vascular redistribution, sometimes with
effusions. Distribution and clinical context distinguish causes.

## [KB-EMPHYSEMA] Emphysema
Permanent airspace enlargement from alveolar wall destruction. Signs:
hyperinflation, flattened diaphragms, increased lucency, and a narrow cardiac
silhouette. Best characterized on CT; chest radiograph is insensitive to early
disease.

## [KB-FIBROSIS] Fibrosis
Scarring with reticular opacities, volume loss, and architectural distortion,
often basal/peripheral. Chronic; comparison with priors is important and CT is
more sensitive.

## [KB-PLEURAL_THICKENING] Pleural Thickening
Thickening of the pleura, which may follow infection, asbestos exposure, or prior
effusion/hemorrhage. Can mimic or coexist with small effusions; calcification
suggests prior asbestos exposure.

## [KB-HERNIA] Hernia (diaphragmatic / hiatal)
Protrusion of abdominal contents into the thorax. May show a retrocardiac
air-fluid level or bowel gas above the diaphragm. The rarest label in the
dataset (very few positives), so model estimates are especially unreliable.
