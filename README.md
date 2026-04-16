# CACTAS-UQ

**Uncertainty Quantification from Calcified Plaque to Outer Wall Estimation in Carotid CTA**
 
This repository implements an ensemble-based uncertainty quantification (UQ) framework for calcified plaque segmentation in carotid CT angiography (CTA) images. Segmentation is performed by **nnU-Net** (residual encoder, 3D full-resolution, 5-fold cross-validation), and voxel-level uncertainty is derived from the ensemble probability maps. The framework extends from reliable plaque boundary detection to exploratory outer wall estimation using variance-based tissue characterization.


---
 
## Background
 
Calcified plaque in carotid artery is a major cause of ischemic stroke. Deep learning methods have shown strong performance for calcified plaque segmentation on CTA. However, the reliability of voxel-level predictions remains underexplored. This work addresses two questions:
 
1. **Model uncertainty (plaque):** Can ensemble disagreement reliably identify segmentation errors at plaque boundaries?
2. **Annotation uncertainty (outer wall):** Can variance-based characterization estimate the vessel wall boundary where manual annotation is unreliable due to low soft-tissue contrast?


---

## Pipeline Overview
![Pipeline](/viz/pipeline.png)

---

## Method Details
 
### Ensemble Uncertainty Quantification
 
Five nnU-Net fold models each produce a voxel-level probability map. The **uncertainty** is computed across the five maps using one of five metrics: 
1. Standard deviation (std)
2. Variance (var)
3. Range: max − min probability across folds
4. Disagreement: fraction of folds disagreeing with the majority vote
5. Mutual information (MI): difference between entropy of the mean and mean of the entropies


### Threshold Selection and Rejection
 
The rejection threshold is selected on **training data** (56 cases). The selected threshold is then **fixed** and applied to the 14 case test set. Voxels with uncertainty above the threshold are rejected, their predictions are excluded from evaluation.
 
### Evaluation
 
- **AUROC**: Measures whether the UQ method can rank errors higher than correct predictions.
- **Rejection metrics**: Dice, IoU, ECE, and Brier score are computed before and after rejection.


### Outer Wall Estimation
 
For the 27 cases with lumen annotations, the wall region is estimated without ground truth:
 
1. Merge lumen and plaque masks
2. Resample to isotropic spacing (0.5 mm), fill holes, dilate by 5 iterations (~2.5 mm)
3. Resample back to original spacing
4. Wall region = dilated mask − (lumen ∪ plaque)
5. Compute local HU variance within the wall region
6. Apply Otsu thresholding to separate high-variance (wall tissue) from low-variance (surrounding tissue)

This is framed as **estimation, not segmentation** — no outer wall ground truth exists.
 

---



 
## Results (Test Set, n=14)
 
**Best method: std** (AUROC = 0.954, threshold = 0.01, coverage = 96.7%)
 
| Metric | Baseline | After Rejection | p-value |
|--------|----------|-----------------|---------|
| Dice | 0.838 | 0.892 | < 0.0001 |
| IoU | 0.726 | 0.812 | < 0.0001 |
| ECE | 0.0166 | 0.0103 | < 0.0001 |
| Brier | 0.0180 | 0.0103 | — |
 
All five methods compared:
 
| Method | Threshold | Coverage | IoU (base→rej) | Dice (base→rej) | AUROC |
|--------|-----------|----------|-----------------|------------------|-------|
| **std** | 0.01 | 0.967 | 0.726 → 0.812 | 0.838 → 0.892 | **0.954** |
| var | 0.01 | 0.973 | 0.726 → 0.792 | 0.838 → 0.879 | 0.954 |
| range | 0.03 | 0.967 | 0.726 → 0.811 | 0.838 → 0.892 | 0.950 |
| disagree | 0.20 | 0.989 | 0.726 → 0.750 | 0.838 → 0.853 | 0.710 |
| MI | 0.01 | 0.967 | 0.726 → 0.811 | 0.838 → 0.892 | 0.863 |
 
---


## Uncertainty Map Example

1. The uncertainty maps highlight and remove ambiguous regions:  
![UQ Map](/viz/uq_map.png)

2. We can compare HU(plaque intensity) map, probability map and uncertainty map:
![inflammogram](/viz/inflammogram.png)

3. Outer wall estimation visualization show data-driven approach estimation:
![outerwall](/viz/outerwall.png)