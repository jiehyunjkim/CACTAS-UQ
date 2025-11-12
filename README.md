# CACTAS-UQ

This repository presents the uncertainty quantification (UQ) pipeline for calcified plaque segmentation in carotid CTA images. Segmentation and probability maps were generated using **nnUNet**, and voxel-level uncertainty was quantified from the probability maps.

---

## Pipeline Overview
![Pipeline](/viz/pipeline.png)

---

## Steps

1. **Compute baseline metrics**  
   - Intersection over Union (IoU)  
   - Expected Calibration Error (ECE)

2. **Apply uncertainty-based rejection**  
   - Recalculate IoU and ECE after removing uncertain voxels.

3. **Report extended performance metrics**  
   - IoU, Dice, F1-score  
   - Confusion matrix components (TP, FP, FN)

4. **Visualize representative plaque cases**  
   - If plaque exists on one side, show a single example.  
   - If plaques exist on both sides, visualize both.

5. **Illustrate voxel-level FP/FN distributions**  
   - Compare before and after uncertainty-based rejection.

---

## Uncertainty Map Example
The uncertainty maps highlight and remove ambiguous regions:  
![UQ Map](/viz/UQ_map.png)
