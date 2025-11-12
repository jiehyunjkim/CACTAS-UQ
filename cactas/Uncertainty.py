import os, glob, warnings
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.stats import entropy, ttest_rel
from sklearn.metrics import confusion_matrix
from skimage import measure
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, PowerNorm, Normalize
import matplotlib as mpl
import json

np.int = int
np.float = float

eps = 1e-8
threshold_fixed = 0.01

class Uncertain:

    def calc_entropy_map(probs):
        H = entropy(probs, axis=-1)
        return H / np.log(probs.shape[-1])  # normalized entropy [0,1]

    def compute_ece(probs, labels, n_bins=10):
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)
        accs = (preds == labels)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            idx = (confs > bins[i]) & (confs <= bins[i + 1])
            if np.any(idx):
                avg_conf = confs[idx].mean()
                avg_acc = accs[idx].mean()
                ece += np.abs(avg_conf - avg_acc) * idx.mean()
        return ece

    def compute_metrix(preds, labels):
        cm = confusion_matrix(labels.flatten(), preds.flatten(), labels=[0, 1])
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        sensitivity = TP / (TP + FN + eps)
        specificity = TN / (TN + FP + eps)
        f1 = 2 * TP / (2 * TP + FP + FN + eps)
        iou = TP / (TP + FP + FN + eps)
        dice = 2 * TP / (2 * TP + FP + FN + eps)
        return {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision,
            "sensitivity": sensitivity, "specificity": specificity,
            "f1": f1, "iou": iou, "dice": dice
        }


    def compute_iou_in_gt_space(pred_nrrd_path, gt_nrrd_path):
        pred = sitk.ReadImage(pred_nrrd_path)
        gt = sitk.ReadImage(gt_nrrd_path)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(gt)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(sitk.Transform())
        pred_resampled = resampler.Execute(pred)
        pred_arr = sitk.GetArrayFromImage(pred_resampled)
        gt_arr = sitk.GetArrayFromImage(gt)
        pred_bin = (pred_arr > 0).astype(np.uint8)
        gt_bin = (gt_arr > 0).astype(np.uint8)
        inter = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        iou = inter / union if union > 0 else 0.0
        print(f"IoU (GT-space): {iou:.4f}  (pred={pred_bin.sum()}  gt={gt_bin.sum()}  intersect={inter})")
        return iou, pred_bin, gt_bin



    def resample_probs_to_gt_space(probs, gt_nrrd_path, pred_nrrd_path=None):
        gt_img = sitk.ReadImage(gt_nrrd_path)
        ref_img = sitk.ReadImage(pred_nrrd_path) if (pred_nrrd_path and os.path.exists(pred_nrrd_path)) else gt_img

        Z, Y, X, C = probs.shape
        probs_gt = np.zeros((gt_img.GetSize()[2], gt_img.GetSize()[1], gt_img.GetSize()[0], C), dtype=np.float32)

        for c in range(C):
            ch = probs[..., c].astype(np.float32)
            ch_img = sitk.GetImageFromArray(ch)
            ch_img.SetSpacing(ref_img.GetSpacing())
            ch_img.SetOrigin(ref_img.GetOrigin())
            ch_img.SetDirection(ref_img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(gt_img)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(sitk.Transform())
            ch_res = resampler.Execute(ch_img)
            probs_gt[..., c] = sitk.GetArrayFromImage(ch_res)

        probs_gt = np.clip(probs_gt, eps, 1 - eps)
        probs_gt /= (probs_gt.sum(axis=-1, keepdims=True) + eps)
        return probs_gt



    def plot_prob_maps_with_contours(image, gt_mask, prob_before, prob_after, case_name, thresh=threshold_fixed):
        cmap_prob = plt.get_cmap("PRGn")
        TP_color, FP_color, FN_color = "blue", "yellow", "red"
        

        labeled = label(gt_mask)
        regions = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
        if len(regions) == 0:
            z_list = [gt_mask.shape[0] // 2]
        elif len(regions) == 1:
            z_list = [int(regions[0].centroid[0])]
        else:
            z_list = [int(r.centroid[0]) for r in regions[:2]]

        n_rows = len(z_list)
        fig, axes_all = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows),
                                     gridspec_kw={'width_ratios': [1, 1, 1]},
                                     constrained_layout=False)
        if n_rows == 1:
            axes_all = [axes_all]

        for row_i, z_idx in enumerate(z_list):
            axes = axes_all[row_i]
            titles = ["CTA + GT", "Before Reject", "After Reject"]

            gt_slice = gt_mask[z_idx]
            y_idx, x_idx = np.where(gt_slice > 0)
            cy, cx = np.mean(y_idx), np.mean(x_idx)
            margin = 10
            y_min, y_max = int(max(0, cy - margin)), int(min(gt_mask.shape[1], cy + margin))
            x_min, x_max = int(max(0, cx - margin)), int(min(gt_mask.shape[2], cx + margin))

            img_crop = image[z_idx, y_min:y_max, x_min:x_max]
            gt_crop = gt_slice[y_min:y_max, x_min:x_max]
            pb_crop = prob_before[z_idx, y_min:y_max, x_min:x_max]
            pa_crop = prob_after[z_idx, y_min:y_max, x_min:x_max]

            img_norm = (img_crop - img_crop.min()) / (img_crop.ptp() + 1e-8)
            # norm = PowerNorm(gamma=0.1, vmin=0, vmax=1.0)
            norm = PowerNorm(gamma=0.1, vmin=0.001, vmax=1.0)
            # norm = Normalize(vmin=0.00, vmax=1.0)

            def get_masks(pred, gt):
                pred_bin = pred > thresh
                tp = pred_bin & gt
                fp = pred_bin & (~gt)
                fn = (~pred_bin) & gt
                return tp, fp, fn

            tp_b, fp_b, fn_b = get_masks(pb_crop, gt_crop)
            tp_a, fp_a, fn_a = get_masks(pa_crop, gt_crop)

            def draw_panel(ax, prob, tp, fp, fn, title):
                ax.imshow(img_norm, cmap="gray")
                im = ax.imshow(prob, cmap=cmap_prob, norm=norm, alpha=0.9)
                tp_mask = np.ma.masked_where(~tp, tp)
                fp_mask = np.ma.masked_where(~fp, fp)
                fn_mask = np.ma.masked_where(~fn, fn)
                ax.imshow(tp_mask, cmap=ListedColormap([TP_color]), alpha=0.3)
                ax.imshow(fp_mask, cmap=ListedColormap([FP_color]), alpha=0.4)
                ax.imshow(fn_mask, cmap=ListedColormap([FN_color]), alpha=0.4)
                for c in measure.find_contours(tp, 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=TP_color, lw=1.3)
                for c in measure.find_contours(fp, 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=FP_color, lw=1.3)
                for c in measure.find_contours(fn, 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=FN_color, lw=1.3)
                ax.set_title(f"{title} — {case_name} (z={z_idx})", fontsize=10)
                ax.axis("off")

            axes[0].imshow(img_norm, cmap="gray")
            gt_overlay = np.zeros_like(gt_crop, dtype=float)
            gt_overlay[gt_crop > 0] = 1.0
            axes[0].imshow(gt_overlay, cmap=cmap_prob, norm=norm, alpha=0.8)
            draw_panel(axes[1], pb_crop, tp_b, fp_b, fn_b, titles[1])
            draw_panel(axes[2], pa_crop, tp_a, fp_a, fn_a, titles[2])

        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap_prob)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Predicted Foreground Probability", fontsize=10)
        plt.subplots_adjust(right=0.9, wspace=0.08, hspace=0.2)
        plt.show()

        
    def load_data(npz_path):
        npz = np.load(npz_path)
        probs = np.moveaxis(npz["probabilities"], 0, -1)
        probs = np.clip(probs, eps, 1 - eps)
        
        return probs
    
    def entropy_map(npz_path, gt_path, probs):
        pred_nrrd_path = npz_path.replace(".npz", ".nrrd")
        base_iou, _, gt_bin = Uncertain.compute_iou_in_gt_space(pred_nrrd_path, gt_path)
        probs_gt = Uncertain.resample_probs_to_gt_space(probs, gt_path, pred_nrrd_path)
        entropy_map = Uncertain.calc_entropy_map(probs_gt)
        
        return base_iou, gt_bin, probs_gt, entropy_map
    
    def uncertainty_calc(probs_gt, entropy_map, gt_bin):
        flat_probs = probs_gt.reshape(-1, probs_gt.shape[-1])
        flat_entropy = entropy_map.flatten()
        flat_labels = gt_bin.flatten()
        flat_preds = np.argmax(flat_probs, axis=-1)
        
        return flat_probs, flat_entropy, flat_labels, flat_preds
        
    
    def calc_ece(probs_gt, gt_bin, base_iou):
        base_ece_all = Uncertain.compute_ece(probs_gt, gt_bin)
        fg_mask = gt_bin > 0
        base_ece_fg = Uncertain.compute_ece(probs_gt[fg_mask], gt_bin[fg_mask]) if fg_mask.sum() > 0 else np.nan
        
        print(f"Base ECE(all)={base_ece_all:.6f}, Base ECE(fg)={base_ece_fg:.6f}, Base IoU={base_iou:.4f}")
        
        return base_ece_all, base_ece_fg
    
    def reject(flat_entropy, flat_preds, flat_labels, flat_probs):
        keep = flat_entropy < threshold_fixed
        iou_reject = Uncertain.compute_metrix(flat_preds[keep], flat_labels[keep])["iou"]
        
        # Calculate Foreground-only ECE 
        fg_mask_flat = (flat_labels > 0)
        keep_fg = keep & fg_mask_flat
        if keep_fg.sum() > 0:
            ece_after_fg = Uncertain.compute_ece(flat_probs[keep_fg], flat_labels[keep_fg])
        else:
            ece_after_fg = np.nan
            
        print(f"After reject (t={threshold_fixed}): ECE(fg)={ece_after_fg:.6f}, IoU={iou_reject:.4f}")
        
        return keep, iou_reject, ece_after_fg
        
    
    def print_metrix(keep, flat_preds, flat_labels):
        before = Uncertain.compute_metrix(flat_preds, flat_labels)
        after = Uncertain.compute_metrix(flat_preds[keep], flat_labels[keep])
        print("\n--- Metrics Comparison ---")
        metrics_list = ["iou", "dice", "f1", "FP", "FN", "TP", "TN"]

        print(f"{'Metric':<15}{'Before':>12}{'After':>12}{'Δ (After-Before)':>20}")
        print("-" * 60)

        for m in metrics_list:
            b = before.get(m, np.nan)
            a = after.get(m, np.nan)
            try:
                diff = float(a) - float(b)
                print(f"{m:<15}{float(b):>12.4f}{float(a):>12.4f}{diff:>20.4f}")
            except Exception:
                print(f"{m:<15}{b:>12}{a:>12}{' ':>20}")
        print("-" * 60)
        
        return before, after
    
    def visualize_uncertainty_map(probs_gt, entropy_map, cta_path, gt_bin, case_num):
        fg_prob_before = probs_gt[..., 1]
        fg_prob_after = fg_prob_before.copy()
        fg_prob_after[entropy_map >= threshold_fixed] = 0
        image = sitk.GetArrayFromImage(sitk.ReadImage(cta_path))

        Uncertain.plot_prob_maps_with_contours(image, gt_bin, fg_prob_before, fg_prob_after, case_name=case_num)
        
        np.savez_compressed(
            f"/raid/mpsych/CACTAS/DATA/nnUNet/Mask/UQ_RESULTS/2D/{case_num}.npz",
            image=image,
            probs_gt=probs_gt,
            fg_prob_before=fg_prob_before,
            fg_prob_after=fg_prob_after,
            entropy_map=entropy_map,
            gt_mask=gt_bin
        )
        print(f"{case_num} case saved!")
        
   
    def visualize_rejection(before, after, case):
        plt.figure(figsize=(6, 4))
        x = np.array([0, 1, 3, 4])
        h = [before["FP"], after["FP"], before["FN"], after["FN"]]
        colors = ['lightgray', 'gray', 'mistyrose', 'indianred']
        labels_bar = ['FP (w/o reject)', 'FP (w/ reject)', 'FN (w/o reject)', 'FN (w/ reject)']
        plt.bar(x, h, color=colors, width=0.8)
        for xi, hi in zip(x, h):
            plt.text(xi, hi + max(1, 0.02 * max(h)), f"{hi}", ha='center', va='bottom', fontsize=9)
        plt.xticks(x, labels_bar, rotation=15)
        plt.ylabel("False Predictions")
        plt.title(f"False Predictions — {case}")
        plt.tight_layout()
        plt.show()

    
    
    def summary(all_base_iou, all_reject_iou, all_ece_before, all_ece_after):
        print("\n================ Summary ================")
        mean_iou_base = np.mean(all_base_iou)
        std_iou_base = np.std(all_base_iou)
        mean_iou_after = np.mean(all_reject_iou)
        std_iou_after = np.std(all_reject_iou)
        mean_ece_base = np.nanmean(all_ece_before)
        std_ece_base = np.nanstd(all_ece_before)
        mean_ece_after = np.nanmean(all_ece_after)
        std_ece_after = np.nanstd(all_ece_after)

        print(f"IoU before reject : {mean_iou_base:.4f} ± {std_iou_base:.4f}")
        print(f"IoU after reject  : {mean_iou_after:.4f} ± {std_iou_after:.4f}")
        print(f"ΔIoU (improvement): {np.mean(np.array(all_reject_iou) - np.array(all_base_iou)):.4f}")
        print(f"ECE before reject (FG): {mean_ece_base:.6f} ± {std_ece_base:.6f}")
        print(f"ECE after reject (FG) : {mean_ece_after:.6f} ± {std_ece_after:.6f}\n")
        
        
        