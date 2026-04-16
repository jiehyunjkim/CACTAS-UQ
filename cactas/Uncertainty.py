import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import generate_binary_structure, binary_dilation as scipy_binary_dilation
from scipy.ndimage import uniform_filter, label as cc_label
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from skimage.filters import threshold_otsu
import mahotas as mh


class Uncertain:

    # ==========================================================
    # CONSTANTS
    # ==========================================================
    FOLDS = [0, 1, 2, 3, 4]
    PROB_THRESH = 0.5
    BAND_RADIUS = 3
    ECE_BINS = 15
    BEST_COV_MIN = 0.95
    REJECT_METHODS = ["std", "var", "range", "disagree", "mi"]
    UQ_THRESHOLDS = [round(x, 2) for x in
                     list(np.arange(0.00, 0.20, 0.01)) + [0.20, 0.30, 0.40, 0.50]]

    REJECT_BY = "std"
    REJECT_T = 0.01

    ISO_SPACING = [0.5, 0.5, 0.5]
    DILATE_ITERS = 5
    VAR_WIN = 7
    OTSU_FACTOR = 1.0

    CT_WL = 120;  CT_WW = 750
    HU_MIN = 120; HU_MAX = 1600; HU_ALPHA = 0.7; HU_CMAP = "turbo"
    WALL_CMAP = "magma"; WALL_ALPHA = 0.65
    CROP_MARGIN = 3; MIN_CROP_SIZE = 24; MIN_COMPONENT_AREA = 30

    # ==========================================================
    # CASE ID MAPPINGS
    # ==========================================================
    NNUNET_TO_ESUS = {
        "044": "2",  "027": "5",  "020": "6",  "066": "7",  "019": "9",
        "003": "10", "040": "11", "029": "12", "004": "13", "026": "14",
        "067": "15", "046": "16", "033": "17", "013": "18", "025": "21",
        "034": "22", "023": "23", "016": "24", "035": "25", "070": "26",
        "069": "27", "045": "29", "051": "31", "048": "32", "009": "33",
        "005": "35", "053": "36", "030": "39", "021": "40", "057": "41",
        "011": "42", "022": "45", "042": "46", "056": "47", "061": "48",
        "064": "49", "054": "50", "052": "51", "063": "52", "065": "53",
        "039": "55", "062": "57", "024": "60", "041": "61", "060": "62",
        "068": "63", "012": "64", "032": "65", "049": "66", "031": "69",
        "058": "71", "036": "72", "017": "73", "006": "75", "018": "77",
        "008": "79", "028": "82", "002": "83", "059": "84", "055": "86",
        "015": "87", "014": "88", "050": "90", "010": "91", "007": "92",
        "038": "93", "001": "94", "043": "95", "047": "96", "037": "97",
    }
    ESUS_TO_NNUNET = {v: k for k, v in NNUNET_TO_ESUS.items()}

    LUMEN_CASES = [
        (57, "062", "test"),  (60, "024", "train"), (61, "041", "train"),
        (62, "060", "test"),  (63, "068", "test"),  (64, "012", "train"),
        (65, "032", "train"), (66, "049", "train"), (69, "031", "train"),
        (72, "036", "train"), (73, "017", "train"), (75, "006", "train"),
        (77, "018", "train"), (79, "008", "train"), (82, "028", "train"),
        (83, "002", "train"), (84, "059", "test"),  (86, "055", "train"),
        (88, "014", "train"), (90, "050", "train"), (91, "010", "train"),
        (92, "007", "train"), (93, "038", "train"), (94, "001", "train"),
        (95, "043", "train"), (96, "047", "train"), (97, "037", "train"),
    ]

    TRAIN_CASE_IDS = [f"{i:03d}" for i in range(1, 57)]
    TEST_CASE_IDS = [f"{i:03d}" for i in range(57, 71)]

    # ==========================================================
    # LOW-LEVEL I/O
    # ==========================================================
    @staticmethod
    def read_vol(path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    @staticmethod
    def read_bool(path):
        return Uncertain.read_vol(path) > 0

    @staticmethod
    def read_sitk(path):
        return sitk.ReadImage(path)

    @staticmethod
    def load_npz_foreground_prob(npz_path):
        z = np.load(npz_path)
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"Empty npz: {npz_path}")
        preferred = ["probabilities", "softmax", "pred", "prediction", "data"]
        arr = None
        for k in preferred:
            if k in z:
                arr = z[k]; break
        if arr is None:
            arr = z[max(keys, key=lambda k: z[k].size)]
        if arr.ndim == 4:
            p_fg = arr[1] if arr.shape[0] >= 2 else arr[0]
        elif arr.ndim == 3:
            p_fg = arr
        else:
            raise ValueError(f"Bad npz shape: {arr.shape}")
        return p_fg.astype(np.float32, copy=False)

    @staticmethod
    def find_nn_file(nn_id, suffix, nnunet_base):
        for sub in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
            p = os.path.join(nnunet_base, sub, f"{nn_id}{suffix}")
            if os.path.isfile(p):
                return p
        return None

    @staticmethod
    def find_esus_cta(esus_id, esus_dir):
        for fmt in [f"{esus_id}.img.nrrd", f"{esus_id}.b.img.nrrd"]:
            p = os.path.join(esus_dir, fmt)
            if os.path.isfile(p):
                return p
        return None

    @staticmethod
    def find_lumen_path(esus_id, elucid_dir):
        for suf in [".ca.nrrd", ".ca.seg.nrrd"]:
            p = os.path.join(elucid_dir, f"{esus_id}{suf}")
            if os.path.isfile(p):
                return p
        return None

    @staticmethod
    def fold_dir(base_output, fold):
        d = f"{base_output}_fold{fold}"
        if not os.path.isdir(d):
            raise FileNotFoundError(d)
        return d

    @staticmethod
    def window_ct(img, wl=120, ww=750):
        lo, hi = wl - ww / 2, wl + ww / 2
        return (np.clip(img, lo, hi) - lo) / (hi - lo + 1e-8)

    @staticmethod
    def robust_clip(img2d, lo_pct=1, hi_pct=99):
        lo = np.percentile(img2d, lo_pct)
        hi = np.percentile(img2d, hi_pct)
        if hi <= lo:
            return img2d
        return (np.clip(img2d, lo, hi) - lo) / (hi - lo + 1e-8)

    # ==========================================================
    # STEP-LEVEL LOADERS (for notebook pipeline)
    # ==========================================================
    @staticmethod
    def read_gt(cid, gt_dir):
        """Load ground truth mask. Returns bool array or None."""
        p = os.path.join(gt_dir, f"{cid}.nrrd")
        if not os.path.isfile(p):
            print(f"Case {cid}: missing GT"); return None
        return Uncertain.read_bool(p)

    @staticmethod
    def read_cta(cid, nnunet_base, esus_dir):
        """Load CTA volume (ESUS original first, then fallback). Returns (cta, esus_id) or (None, None)."""
        U = Uncertain
        esus_id = U.NNUNET_TO_ESUS.get(cid)
        path = None
        if esus_id:
            path = U.find_esus_cta(esus_id, esus_dir)
        if path is None:
            path = U.find_nn_file(cid, "_0000.nrrd", nnunet_base)
        if path is None:
            print(f"Case {cid}: missing CTA"); return None, None
        return U.read_vol(path).astype(np.float32), esus_id

    @staticmethod
    def load_prob_stack(cid, prob_base):
        """Load 5-fold probability maps. Returns (5,Z,H,W) stack or None."""
        U = Uncertain
        fold_dirs = {f: U.fold_dir(prob_base, f) for f in U.FOLDS}
        p_list = []
        for f in U.FOLDS:
            p_path = os.path.join(fold_dirs[f], f"{cid}.npz")
            if not os.path.isfile(p_path):
                print(f"Case {cid}: missing prob fold {f}"); return None
            p_list.append(U.load_npz_foreground_prob(p_path))
        return np.stack(p_list, axis=0)

    @staticmethod
    def build_rejection(gt, maps):
        """From GT + UQ maps, build pred, eval_mask, uq, reject.
        Returns (pred, eval_mask, uq, reject)."""
        U = Uncertain
        pred = maps["p_mean"] >= U.PROB_THRESH
        eval_mask = U.build_eval_mask(gt, U.BAND_RADIUS)
        uq = maps[U.REJECT_BY]
        reject = np.zeros_like(eval_mask, dtype=bool)
        reject[eval_mask] = uq[eval_mask] > U.REJECT_T
        return pred, eval_mask, uq, reject

    @staticmethod
    def read_wall_cta(esus_id, nn_id, nnunet_base, esus_dir):
        """Load CTA for wall analysis. Returns (cta, cta_sitk) or (None, None)."""
        U = Uncertain
        path = U.find_esus_cta(str(esus_id), esus_dir)
        if path is None:
            path = U.find_nn_file(nn_id, "_0000.nrrd", nnunet_base)
            if path:
                print(f"  Fallback to masked CTA")
        if path is None:
            print(f"  ESUS {esus_id}: missing CTA"); return None, None
        cta_sitk = U.read_sitk(path)
        cta = sitk.GetArrayFromImage(cta_sitk).astype(np.float32)
        return cta, cta_sitk

    @staticmethod
    def read_plaque_gt(nn_id, nnunet_base):
        """Load plaque GT mask. Returns bool array or None."""
        path = Uncertain.find_nn_file(nn_id, ".nrrd", nnunet_base)
        if path is None:
            print(f"  nn{nn_id}: missing plaque GT"); return None
        return Uncertain.read_bool(path)

    @staticmethod
    def read_lumen(esus_id, elucid_dir):
        """Load lumen mask from ELUCID. Returns bool array or None."""
        path = Uncertain.find_lumen_path(esus_id, elucid_dir)
        if path is None:
            print(f"  ESUS {esus_id}: missing lumen"); return None
        return Uncertain.read_bool(path)

    # ==========================================================
    # MASK OPS
    # ==========================================================
    @staticmethod
    def binary_dilate(mask_bool, radius):
        if radius <= 0:
            return mask_bool
        img = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
        dil = sitk.BinaryDilate(img, [radius] * 3)
        return sitk.GetArrayFromImage(dil).astype(bool)

    @staticmethod
    def build_eval_mask(gt, band_radius=3):
        return Uncertain.binary_dilate(gt, band_radius)

    @staticmethod
    def apply_rejection(pred_bool, reject_mask):
        out = pred_bool.copy()
        out[reject_mask] = False
        return out

    # ==========================================================
    # UQ COMPUTATION
    # ==========================================================
    @staticmethod
    def entropy_binary(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    @staticmethod
    def compute_uq_maps(stack_pfg):
        """Compute all UQ metrics from 5-fold probability stack."""
        p_mean = np.mean(stack_pfg, axis=0)
        p_std = np.std(stack_pfg, axis=0)
        p_var = np.var(stack_pfg, axis=0)
        p_range = np.max(stack_pfg, axis=0) - np.min(stack_pfg, axis=0)
        hard = stack_pfg >= 0.5
        maj = np.mean(hard, axis=0) >= 0.5
        disagree = np.mean(hard != maj, axis=0).astype(np.float32)
        H_mean = Uncertain.entropy_binary(p_mean)
        H_each = np.mean(Uncertain.entropy_binary(stack_pfg), axis=0)
        mi = np.maximum(0.0, H_mean - H_each).astype(np.float32)
        return {
            "p_mean": p_mean.astype(np.float32),
            "std": p_std.astype(np.float32),
            "var": p_var.astype(np.float32),
            "range": p_range.astype(np.float32),
            "disagree": disagree,
            "mi": mi,
        }

    # ==========================================================
    # METRICS
    # ==========================================================
    @staticmethod
    def confusion_counts(y_true, y_pred):
        tp = int(np.sum(y_true & y_pred))
        tn = int(np.sum(~y_true & ~y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        return tp, tn, fp, fn

    @staticmethod
    def iou_dice(tp, fp, fn):
        d_iou = tp + fp + fn
        iou = (tp / d_iou) if d_iou > 0 else float("nan")
        d_dice = 2 * tp + fp + fn
        dice = (2 * tp / d_dice) if d_dice > 0 else float("nan")
        return float(iou), float(dice)

    @staticmethod
    def ece_binary(p_fg, y_true, n_bins=15):
        p = p_fg.astype(np.float32, copy=False)
        y = y_true.astype(bool, copy=False)
        pred = p >= 0.5
        conf = np.where(pred, p, 1.0 - p).astype(np.float32)
        correct = (pred == y).astype(np.float32)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        N = conf.size
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            m = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
            cnt = int(np.sum(m))
            if cnt == 0:
                continue
            ece += (cnt / N) * abs(float(np.mean(correct[m])) - float(np.mean(conf[m])))
        return float(ece)

    @staticmethod
    def brier_binary(p_fg, y_true):
        return float(np.mean((p_fg - y_true.astype(np.float32)) ** 2))

    @staticmethod
    def error_detection_auroc(uq_vals, y_true, y_pred):
        error = (y_true != y_pred).astype(int)
        if error.sum() == 0 or error.sum() == error.size:
            return float("nan"), float("nan")
        mask = np.isfinite(uq_vals)
        if not mask.all():
            uq_vals, error = uq_vals[mask], error[mask]
        if error.sum() == 0 or error.sum() == error.size:
            return float("nan"), float("nan")
        return float(roc_auc_score(error, uq_vals)), float(average_precision_score(error, uq_vals))

    @staticmethod
    def nanmean(x):
        a = np.asarray(x, dtype=float)
        return float(np.nanmean(a)) if a.size > 0 else float("nan")

    # ==========================================================
    # TASK 1: BULK LOADER + THRESHOLD + EVAL + REPORTING
    # ==========================================================
    @staticmethod
    def load_uq_case(cid, gt_dir, fold_dirs):
        """Load one case for UQ evaluation (used by load_all_cases)."""
        U = Uncertain
        gt_path = os.path.join(gt_dir, f"{cid}.nrrd")
        if not os.path.isfile(gt_path):
            return None
        gt = U.read_bool(gt_path)
        p_list = []
        for f in U.FOLDS:
            p_path = os.path.join(fold_dirs[f], f"{cid}.npz")
            if not os.path.isfile(p_path):
                return None
            p_list.append(U.load_npz_foreground_prob(p_path))
        stack = np.stack(p_list, axis=0)
        maps = U.compute_uq_maps(stack)
        p_mean = maps["p_mean"]
        pred = p_mean >= U.PROB_THRESH
        eval_mask = U.build_eval_mask(gt, U.BAND_RADIUS)
        eval_vox = int(np.sum(eval_mask))
        if eval_vox == 0:
            return None
        return {"cid": cid, "gt": gt, "p_mean": p_mean, "pred": pred,
                "eval_mask": eval_mask, "eval_vox": eval_vox, "maps": maps}

    @staticmethod
    def load_all_cases(gt_dir, prob_base, split="train"):
        """Load all train or test cases for UQ evaluation."""
        U = Uncertain
        case_ids = U.TRAIN_CASE_IDS if split == "train" else U.TEST_CASE_IDS
        fold_dirs = {f: U.fold_dir(prob_base, f) for f in U.FOLDS}
        print(f"Loading {split} cases...")
        cases = []
        for cid in case_ids:
            pack = U.load_uq_case(cid, gt_dir, fold_dirs)
            if pack is not None:
                cases.append(pack)
        print(f"  Loaded {len(cases)} {split} cases\n")
        return cases

    @staticmethod
    def select_threshold(method, cases):
        """Pick best UQ rejection threshold from training data."""
        U = Uncertain
        agg = {t: {"cov": [], "iou": [], "dice": []} for t in U.UQ_THRESHOLDS}
        for pack in cases:
            gt, pred, eval_mask, eval_vox = pack["gt"], pack["pred"], pack["eval_mask"], pack["eval_vox"]
            uq = pack["maps"][method]
            for t in U.UQ_THRESHOLDS:
                reject = np.zeros_like(eval_mask, dtype=bool)
                reject[eval_mask] = uq[eval_mask] > t
                kept = eval_mask & ~reject
                cov = float(np.sum(kept) / max(1, eval_vox))
                pred_t = U.apply_rejection(pred, reject)
                tp, _, fp, fn = U.confusion_counts(gt[kept], pred_t[kept])
                iou, dice = U.iou_dice(tp, fp, fn)
                agg[t]["cov"].append(cov)
                agg[t]["iou"].append(iou)
                agg[t]["dice"].append(dice)
        best_t, best_score = None, -1e9
        for t in U.UQ_THRESHOLDS:
            cov_m = U.nanmean(agg[t]["cov"])
            if cov_m >= U.BEST_COV_MIN:
                score = 0.5 * (U.nanmean(agg[t]["iou"]) + U.nanmean(agg[t]["dice"]))
                if score > best_score:
                    best_score = score
                    best_t = t
        return best_t

    @staticmethod
    def evaluate_test(method, best_t, cases):
        """Evaluate on test set. Prints aggregate + t-test. Returns results dict."""
        U = Uncertain
        results = {k: [] for k in [
            "base_iou", "base_dice", "base_fn", "base_fp", "base_ece", "base_brier",
            "rej_iou", "rej_dice", "rej_fn", "rej_fp", "rej_ece", "rej_brier",
            "rej_cov", "auroc", "auprc"]}
        all_uq, all_err = [], []
        for pack in cases:
            gt, pred, p_mean = pack["gt"], pack["pred"], pack["p_mean"]
            eval_mask, eval_vox = pack["eval_mask"], pack["eval_vox"]
            uq = pack["maps"][method]
            y_eval, pred_eval, p_eval = gt[eval_mask], pred[eval_mask], p_mean[eval_mask]
            tp0, _, fp0, fn0 = U.confusion_counts(y_eval, pred_eval)
            b_iou, b_dice = U.iou_dice(tp0, fp0, fn0)
            b_ece = U.ece_binary(p_eval, y_eval)
            b_brier = U.brier_binary(p_eval, y_eval)
            reject = np.zeros_like(eval_mask, dtype=bool)
            reject[eval_mask] = uq[eval_mask] > best_t
            kept = eval_mask & ~reject
            cov = float(np.sum(kept) / max(1, eval_vox))
            pred_t = U.apply_rejection(pred, reject)
            tp, _, fp, fn = U.confusion_counts(gt[kept], pred_t[kept])
            r_iou, r_dice = U.iou_dice(tp, fp, fn)
            p_kept = p_mean[kept]
            r_ece = U.ece_binary(p_kept, gt[kept]) if p_kept.size > 0 else float("nan")
            r_brier = U.brier_binary(p_kept, gt[kept]) if p_kept.size > 0 else float("nan")
            uq_eval = uq[eval_mask]
            auroc, auprc = U.error_detection_auroc(uq_eval, y_eval, pred_eval)
            all_uq.append(uq_eval)
            all_err.append((y_eval != pred_eval).astype(int))
            for k, v in [("base_iou", b_iou), ("base_dice", b_dice), ("base_fn", fn0),
                         ("base_fp", fp0), ("base_ece", b_ece), ("base_brier", b_brier),
                         ("rej_iou", r_iou), ("rej_dice", r_dice), ("rej_fn", fn),
                         ("rej_fp", fp), ("rej_ece", r_ece), ("rej_brier", r_brier),
                         ("rej_cov", cov), ("auroc", auroc), ("auprc", auprc)]:
                results[k].append(v)
        results["all_uq"] = np.concatenate(all_uq)
        results["all_error"] = np.concatenate(all_err)
        print(f"\n===== {method} (t={best_t}) =====")
        for m in ["iou", "dice", "fn", "fp", "ece", "brier"]:
            print(f"  {m:<6s}  {U.nanmean(results[f'base_{m}']):10.4f} -> {U.nanmean(results[f'rej_{m}']):10.4f}")
        print(f"  cov    {U.nanmean(results['rej_cov']):10.4f}")
        print(f"  AUROC  {U.nanmean(results['auroc']):10.4f}")
        t_iou, p_iou = ttest_rel(results["base_iou"], results["rej_iou"])
        t_dice, p_dice = ttest_rel(results["base_dice"], results["rej_dice"])
        t_ece, p_ece = ttest_rel(results["base_ece"], results["rej_ece"])
        print(f"  t-test IoU: t={t_iou:.2f}, p={p_iou:.4f}")
        print(f"  t-test Dice: t={t_dice:.2f}, p={p_dice:.4f}")
        print(f"  t-test ECE: t={t_ece:.2f}, p={p_ece:.4f}")
        return results

    @staticmethod
    def comparison_table(summary):
        nm = Uncertain.nanmean
        print("\n" + "=" * 100)
        print(f"{'method':<10} {'best_t':>6}  {'cov':>7}  {'iou_base->rej':>15}  {'dice_base->rej':>16}  "
              f"{'fn_base->rej':>14}  {'fp_base->rej':>14}  {'ece_base->rej':>15}  {'brier_base->rej':>17}  {'auroc':>6}")
        for method, best_t, results in summary:
            if results is None:
                print(f"{method:<10}  --"); continue
            print(f"{method:<10} {best_t:6.2f}  {nm(results['rej_cov']):7.4f}  "
                  f"{nm(results['base_iou']):7.4f}->{nm(results['rej_iou']):7.4f}  "
                  f"{nm(results['base_dice']):8.4f}->{nm(results['rej_dice']):7.4f}  "
                  f"{nm(results['base_fn']):7.1f}->{nm(results['rej_fn']):5.1f}  "
                  f"{nm(results['base_fp']):7.1f}->{nm(results['rej_fp']):5.1f}  "
                  f"{nm(results['base_ece']):7.5f}->{nm(results['rej_ece']):7.5f}  "
                  f"{nm(results['base_brier']):8.5f}->{nm(results['rej_brier']):8.5f}  "
                  f"{nm(results['auroc']):6.4f}")

    @staticmethod
    def best_method(summary):
        nm = Uncertain.nanmean
        valid = [(m, t, r) for m, t, r in summary if r is not None]
        if not valid:
            print("No valid methods found."); return None
        best = max(valid, key=lambda x: (nm(x[2]["auroc"]), nm(x[2]["rej_dice"]), -nm(x[2]["rej_ece"])))
        print(f"\nBest: {best[0]} (AUROC={nm(best[2]['auroc']):.4f})")
        return best

    @staticmethod
    def plot_auroc_curves(summary):
        U = Uncertain
        colors = {"std": "blue", "var": "red", "range": "green", "disagree": "orange", "mi": "purple"}
        fig, ax = plt.subplots(figsize=(8, 7))
        for method, best_t, results in summary:
            if results is None or "all_uq" not in results:
                continue
            fpr, tpr, _ = roc_curve(results["all_error"], results["all_uq"])
            ax.plot(fpr, tpr, color=colors.get(method, "gray"), linewidth=2,
                    label=f"{method} (AUROC={U.nanmean(results['auroc']):.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Error Detection"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    # ==========================================================
    # GEOMETRY: PLAQUE SIDES + CROP
    # ==========================================================
    @staticmethod
    def find_plaque_sides(mask3d):
        Z, H, W = mask3d.shape
        mid = W // 2
        left = mask3d.copy(); left[:, :, mid:] = False
        right = mask3d.copy(); right[:, :, :mid] = False
        sides = []
        if np.any(left):
            zs = np.where(np.sum(left, axis=(1, 2)) > 0)[0]
            sides.append({"side": "left", "mask3d": left, "slices": zs.tolist()})
        if np.any(right):
            zs = np.where(np.sum(right, axis=(1, 2)) > 0)[0]
            sides.append({"side": "right", "mask3d": right, "slices": zs.tolist()})
        if not sides:
            zs = np.where(np.sum(mask3d, axis=(1, 2)) > 0)[0]
            if zs.size > 0:
                sides.append({"side": "center", "mask3d": mask3d, "slices": zs.tolist()})
        return sides

    @staticmethod
    def bbox_from_mask_2d(mask2d):
        ys, xs = np.where(mask2d)
        if ys.size == 0: return None
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    @staticmethod
    def expand_bbox_square(bbox, H, W, margin, min_size):
        y0, y1, x0, x1 = bbox
        y0 = max(0, y0 - margin); y1 = min(H - 1, y1 + margin)
        x0 = max(0, x0 - margin); x1 = min(W - 1, x1 + margin)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = max((y1 - y0 + 1) // 2, (x1 - x0 + 1) // 2, min_size // 2)
        y0 = max(0, cy - half); y1 = min(H - 1, cy + half)
        x0 = max(0, cx - half); x1 = min(W - 1, cx + half)
        return y0, y1 + 1, x0, x1 + 1

    @staticmethod
    def get_consistent_crop(side_mask3d, H, W, margin, min_size):
        union_2d = np.any(side_mask3d, axis=0)
        bbox = Uncertain.bbox_from_mask_2d(union_2d)
        if bbox is None: return (0, H, 0, W)
        return Uncertain.expand_bbox_square(bbox, H, W, margin, min_size)

    @staticmethod
    def zoom_box_from_mask(mask2d, pad, min_r, max_r, H, W):
        ys, xs = np.where(mask2d)
        if ys.size == 0: return (0, H, 0, W)
        y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
        x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
        s = max(y1 - y0, x1 - x0)
        cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
        r = max(min_r, min(max_r, s // 2))
        y0, x0 = int(round(cy - r)), int(round(cx - r))
        y1, x1 = y0 + 2 * r, x0 + 2 * r
        if y0 < 0: y1 -= y0; y0 = 0
        if x0 < 0: x1 -= x0; x0 = 0
        if y1 > H: y0 -= (y1 - H); y1 = H
        if x1 > W: x0 -= (x1 - W); x1 = W
        return (max(0, y0), min(H, y1), max(0, x0), min(W, x1))

    @staticmethod
    def find_components_2d(mask2d, min_area=30):
        labeled, n = cc_label(mask2d.astype(np.uint8))
        comps = []
        for cid in range(1, n + 1):
            comp = (labeled == cid)
            area = int(np.sum(comp))
            if area < min_area: continue
            ys, xs = np.where(comp)
            bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            comps.append({"mask": comp, "bbox": bbox, "area": area})
        comps.sort(key=lambda c: c["area"], reverse=True)
        return comps

    # ==========================================================
    # ISOTROPIC DILATION (outer wall)
    # ==========================================================
    @staticmethod
    def _sitk_to_xyz(img):
        return sitk.GetArrayFromImage(img).transpose(2, 1, 0)

    @staticmethod
    def _xyz_to_sitk(arr_xyz, spacing, ref=None):
        img = sitk.GetImageFromArray(arr_xyz.transpose(2, 0, 1))
        img.SetSpacing(spacing)
        if ref:
            img.SetOrigin(ref.GetOrigin()); img.SetDirection(ref.GetDirection())
        return img

    @staticmethod
    def _resample(img_in, out_spacing, interp=sitk.sitkNearestNeighbor):
        sp, sz = img_in.GetSpacing(), img_in.GetSize()
        sz_out = [int(round(s * p / o)) for s, p, o in zip(sz, sp, out_spacing)]
        f = sitk.ResampleImageFilter()
        f.SetOutputSpacing(out_spacing); f.SetSize(sz_out); f.SetInterpolator(interp)
        f.SetOutputOrigin(img_in.GetOrigin()); f.SetOutputDirection(img_in.GetDirection())
        return f.Execute(img_in)

    @staticmethod
    def isotropic_dilate(mask_zyx, ref_sitk, iso_spacing, iters):
        U = Uncertain
        mask_xyz = mask_zyx.transpose(2, 1, 0).astype(np.uint8)
        mask_img = U._xyz_to_sitk(mask_xyz, ref_sitk.GetSpacing(), ref=ref_sitk)
        mask_iso = U._resample(mask_img, iso_spacing)
        arr_iso = U._sitk_to_xyz(mask_iso).astype(np.uint8)
        filled = np.zeros_like(arr_iso, dtype=np.uint8)
        for z in range(arr_iso.shape[2]):
            filled[:, :, z] = mh.close_holes(arr_iso[:, :, z].astype(bool)).astype(np.uint8)
        struct = generate_binary_structure(3, 1)
        dil = scipy_binary_dilation(filled.astype(bool), structure=struct, iterations=iters).astype(np.uint8)
        dil_img = U._xyz_to_sitk(dil, iso_spacing, ref=ref_sitk)
        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(ref_sitk); f.SetInterpolator(sitk.sitkNearestNeighbor)
        return sitk.GetArrayFromImage(f.Execute(dil_img)).astype(bool)

    # ==========================================================
    # VARIANCE + OTSU (outer wall)
    # ==========================================================
    @staticmethod
    def local_variance_2d(img2d, win=7):
        img = img2d.astype(np.float32)
        mean = uniform_filter(img, size=win)
        mean2 = uniform_filter(img * img, size=win)
        return np.maximum(mean2 - mean * mean, 0.0)

    @staticmethod
    def otsu_wall(var_2d, wall_region_2d, factor=1.0):
        vals = var_2d[wall_region_2d]
        if vals.size < 10:
            return np.zeros_like(wall_region_2d, dtype=bool), 0.0
        try:
            thresh = threshold_otsu(vals) * factor
        except ValueError:
            return np.zeros_like(wall_region_2d, dtype=bool), 0.0
        return wall_region_2d & (var_2d >= thresh), float(thresh)

    # ==========================================================
    # DRAWING (per-slice)
    # ==========================================================
    @staticmethod
    def draw_uq_rejection(cta_norm, gt_m, pr_m, uq_m, ev_m, rj_m, title):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(title, fontsize=11, y=1.02)
        axes[0].imshow(cta_norm, cmap="gray"); axes[0].set_title("CTA"); axes[0].axis("off")
        axes[1].imshow(cta_norm, cmap="gray")
        if np.any(pr_m): axes[1].contour(pr_m.astype(float), levels=[0.5], colors="magenta", linewidths=1.2)
        if np.any(gt_m): axes[1].contour(gt_m.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        axes[1].set_title("Pred(m) + GT(w)"); axes[1].axis("off")
        fp = pr_m & ~gt_m; kept = ev_m & ~rj_m; pr_kept = pr_m & ~rj_m; fp_kept = pr_kept & ~gt_m
        uq_before = np.zeros_like(uq_m, dtype=np.float32); uq_before[ev_m] = uq_m[ev_m]
        axes[2].imshow(uq_before, cmap="viridis")
        if np.any(rj_m): axes[2].imshow(np.ma.masked_where(~rj_m, rj_m.astype(float)), cmap="Wistia", alpha=0.55)
        if np.any(pr_m): axes[2].contour(pr_m.astype(float), levels=[0.5], colors="magenta", linewidths=1)
        if np.any(gt_m): axes[2].contour(gt_m.astype(float), levels=[0.5], colors="white", linewidths=1)
        if np.any(fp): axes[2].contour(fp.astype(float), levels=[0.5], colors="red", linewidths=1)
        axes[2].set_title("UQ before rejection"); axes[2].axis("off")
        uq_after = np.zeros_like(uq_m, dtype=np.float32); uq_after[kept] = uq_m[kept]
        axes[3].imshow(uq_after, cmap="viridis")
        if np.any(pr_kept): axes[3].contour(pr_kept.astype(float), levels=[0.5], colors="magenta", linewidths=1)
        if np.any(gt_m): axes[3].contour(gt_m.astype(float), levels=[0.5], colors="white", linewidths=1)
        if np.any(fp_kept): axes[3].contour(fp_kept.astype(float), levels=[0.5], colors="red", linewidths=1)
        axes[3].set_title("UQ after rejection"); axes[3].axis("off")
        plt.tight_layout(); plt.show(); plt.close(fig)

    @staticmethod
    def draw_combined(ct_raw, gt_c, pred_c, pmean_c, uq_c, eval_c, title, reject_by="std"):
        U = Uncertain
        ct_win = U.window_ct(ct_raw, U.CT_WL, U.CT_WW)
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        fig.suptitle(title, fontsize=11, y=1.02)
        axes[0].imshow(ct_win, cmap="gray")
        if np.any(gt_c): axes[0].contour(gt_c.astype(float), levels=[0.5], colors="cyan", linewidths=1)
        axes[0].set_title("CTA + GT contour"); axes[0].axis("off")
        im0 = axes[0].imshow(ct_win, cmap="gray")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label("HU")
        base_rgb = np.dstack([ct_win] * 3)
        norm_hu = Normalize(vmin=U.HU_MIN, vmax=U.HU_MAX, clip=True)
        cmap_hu = cm.get_cmap(U.HU_CMAP)
        color = cmap_hu(norm_hu(ct_raw.astype(np.float32)))[..., :3]
        overlay = base_rgb.copy()
        if np.any(gt_c):
            m = gt_c.astype(bool); overlay[m] = (1 - U.HU_ALPHA) * overlay[m] + U.HU_ALPHA * color[m]
        axes[1].imshow(overlay)
        if np.any(gt_c): axes[1].contour(gt_c.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        axes[1].set_title("Inflammogram (HU)"); axes[1].axis("off")
        sm = cm.ScalarMappable(norm=norm_hu, cmap=cmap_hu); sm.set_array([])
        fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04).set_label("HU")
        prob_vis = np.zeros_like(pmean_c); prob_vis[eval_c] = pmean_c[eval_c]
        axes[2].imshow(ct_win, cmap="gray", alpha=0.3)
        im_c = axes[2].imshow(prob_vis, cmap="magma", vmin=0, vmax=1, alpha=0.85)
        if np.any(gt_c): axes[2].contour(gt_c.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        if np.any(pred_c): axes[2].contour(pred_c.astype(float), levels=[0.5], colors="magenta", linewidths=0.8)
        axes[2].set_title("Prob map (cyan=GT, m=pred)"); axes[2].axis("off")
        fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04).set_label("P(plaque)")
        uq_vis = np.zeros_like(uq_c, dtype=np.float32); uq_vis[eval_c] = uq_c[eval_c]
        im_d = axes[3].imshow(uq_vis, cmap="viridis")
        if np.any(pred_c): axes[3].contour(pred_c.astype(float), levels=[0.5], colors="magenta", linewidths=0.8)
        if np.any(gt_c): axes[3].contour(gt_c.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        axes[3].set_title(f"UQ ({reject_by})"); axes[3].axis("off")
        fig.colorbar(im_d, ax=axes[3], fraction=0.046, pad=0.04).set_label(f"UQ ({reject_by})")
        plt.tight_layout(); plt.show(); plt.close(fig)

    @staticmethod
    def draw_wall(ct_raw, ct_win, lumen, plaque, wall_region, est_outer,
                  var_2d, high_var, otsu_thresh, bbox, title):
        U = Uncertain
        y0, y1, x0, x1 = bbox
        ct_c = ct_win[y0:y1, x0:x1]
        lu_c = lumen[y0:y1, x0:x1]; pl_c = plaque[y0:y1, x0:x1]
        wall_c = wall_region[y0:y1, x0:x1]; ow_c = est_outer[y0:y1, x0:x1]
        var_c = var_2d[y0:y1, x0:x1]; hv_c = high_var[y0:y1, x0:x1]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5)); fig.suptitle(title, fontsize=10)
        axes[0].imshow(ct_c, cmap="gray")
        if np.any(lu_c): axes[0].contour(lu_c.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
        if np.any(pl_c): axes[0].contour(pl_c.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        if np.any(ow_c): axes[0].contour(ow_c.astype(float), levels=[0.5], colors="red", linewidths=1, linestyles="dashed")
        if np.any(hv_c): axes[0].contour(hv_c.astype(float), levels=[0.5], colors="yellow", linewidths=1.2)
        axes[0].set_title("lumen(cyan) plaque(lime)\ndilation(red) Otsu(yellow)"); axes[0].axis("off")
        sm0 = plt.cm.ScalarMappable(cmap="gray"); sm0.set_array([])
        cb0 = fig.colorbar(sm0, ax=axes[0], fraction=0.046, pad=0.04); cb0.ax.set_visible(False)
        var_ring = var_c.copy().astype(np.float32); var_ring[~wall_c] = np.nan
        axes[1].imshow(ct_c, cmap="gray")
        vals = var_ring[~np.isnan(var_ring)]
        if vals.size > 0:
            vmin, vmax = np.percentile(vals, [5, 95])
            im = axes[1].imshow(var_ring, cmap=U.WALL_CMAP, alpha=U.WALL_ALPHA, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label(f"local var (w={U.VAR_WIN})")
        if np.any(lu_c): axes[1].contour(lu_c.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        if np.any(pl_c): axes[1].contour(pl_c.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
        axes[1].set_title(f"Wall variance\nOtsu={otsu_thresh:.0f}"); axes[1].axis("off")
        vis = np.zeros((*ct_c.shape, 3), dtype=np.float32)
        for ch in range(3): vis[:, :, ch] = ct_c
        low = wall_c & ~hv_c
        if np.any(low): vis[low] = vis[low] * 0.4 + np.array([0.2, 0.3, 0.8]) * 0.6
        if np.any(hv_c): vis[hv_c] = vis[hv_c] * 0.4 + np.array([0.9, 0.5, 0.1]) * 0.6
        axes[2].imshow(vis)
        if np.any(lu_c): axes[2].contour(lu_c.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        if np.any(pl_c): axes[2].contour(pl_c.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
        axes[2].set_title("Otsu: wall(orange)\noutside(blue)"); axes[2].axis("off")
        sm2 = plt.cm.ScalarMappable(cmap="gray"); sm2.set_array([])
        cb2 = fig.colorbar(sm2, ax=axes[2], fraction=0.046, pad=0.04); cb2.ax.set_visible(False)
        var_ring2 = var_c.copy().astype(np.float32); var_ring2[~wall_c] = np.nan
        axes[3].imshow(ct_c, cmap="gray")
        vals2 = var_ring2[~np.isnan(var_ring2)]
        if vals2.size > 0:
            vmin, vmax = np.percentile(vals2, [5, 95])
            im2 = axes[3].imshow(var_ring2, cmap=U.WALL_CMAP, alpha=U.WALL_ALPHA, vmin=vmin, vmax=vmax)
            fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04).set_label("local var")
        if np.any(lu_c): axes[3].contour(lu_c.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        if np.any(pl_c): axes[3].contour(pl_c.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
        owc = hv_c | lu_c | pl_c
        if np.any(owc): axes[3].contour(owc.astype(float), levels=[0.5], colors="yellow", linewidths=1.2)
        axes[3].set_title("Variance + Otsu boundary"); axes[3].axis("off")
        plt.tight_layout(); plt.show(); plt.close(fig)

    # ==========================================================
    # PER-CASE VIZ DRIVERS (slice loop + crop + draw)
    # ==========================================================
    @staticmethod
    def viz_plaque_slices(cid, esus_id, cta, gt, pred, uq, eval_mask, reject):
        """Loop through slices and draw UQ before/after rejection."""
        U = Uncertain
        Z, H, W = gt.shape
        sides = U.find_plaque_sides(gt)
        if not sides: sides = U.find_plaque_sides(pred)
        if not sides:
            print(f"Case {cid}: no plaque"); return
        print(f"Case {cid} (ESUS {esus_id}): {len(sides)} side(s)")
        for s in sides:
            union_2d = np.any(s["mask3d"], axis=0)
            ya, yb, xa, xb = U.zoom_box_from_mask(union_2d, 8, 12, 60, H, W)
            for z in s["slices"]:
                gt_c = gt[z, ya:yb, xa:xb]; pr_c = pred[z, ya:yb, xa:xb]
                if not np.any(gt_c) and not np.any(pr_c): continue
                ev_c = eval_mask[z, ya:yb, xa:xb]; rj_c = reject[z, ya:yb, xa:xb]
                uq_c = uq[z, ya:yb, xa:xb]; cta_c = U.robust_clip(cta[z, ya:yb, xa:xb])
                cov = float(np.sum(ev_c & ~rj_c) / max(1, np.sum(ev_c)))
                title = f"Case {cid} | {s['side']} | z={z} | {U.REJECT_BY} t={U.REJECT_T} | cov={cov:.3f}"
                U.draw_uq_rejection(cta_c, gt_c, pr_c, uq_c, ev_c, rj_c, title)

    @staticmethod
    def viz_combined_slices(cid, esus_id, cta, gt, pred, p_mean, uq, eval_mask):
        """Loop through slices and draw CTA + inflammogram + prob + UQ."""
        U = Uncertain
        Z, H, W = gt.shape
        sides = U.find_plaque_sides(gt)
        if not sides: sides = U.find_plaque_sides(pred)
        if not sides:
            print(f"Case {cid}: no plaque"); return
        print(f"Case {cid} (ESUS {esus_id}): {len(sides)} side(s)")
        for s in sides:
            y0, y1, x0, x1 = U.get_consistent_crop(s["mask3d"], H, W, U.CROP_MARGIN, U.MIN_CROP_SIZE)
            for z in s["slices"]:
                gt_c = gt[z, y0:y1, x0:x1]; pr_c = pred[z, y0:y1, x0:x1]
                if not np.any(gt_c) and not np.any(pr_c): continue
                title = f"Case {cid} | {s['side']} | z={z} | {U.REJECT_BY} t={U.REJECT_T}"
                U.draw_combined(cta[z, y0:y1, x0:x1], gt_c, pr_c,
                                p_mean[z, y0:y1, x0:x1], uq[z, y0:y1, x0:x1],
                                eval_mask[z, y0:y1, x0:x1], title, U.REJECT_BY)

    @staticmethod
    def viz_wall_slices(esus_id, nn_id, cta, lumen, plaque, est_outer, wall_region):
        """Loop through lumen+plaque slices: local variance -> Otsu -> draw."""
        U = Uncertain
        Z, H, W = cta.shape
        z_slices = [z for z in np.where(np.sum(lumen, axis=(1, 2)) > 0)[0]
                    if np.any(plaque[z])]
        for z in z_slices:
            ct_win = U.window_ct(cta[z], U.CT_WL, U.CT_WW)
            var_2d = U.local_variance_2d(cta[z], win=U.VAR_WIN)
            visible = lumen[z] | plaque[z]
            comps = U.find_components_2d(visible, U.MIN_COMPONENT_AREA)
            for comp in comps:
                comp_dil = scipy_binary_dilation(
                    comp["mask"], iterations=U.DILATE_ITERS + 2).astype(bool)
                comp_wall = wall_region[z] & comp_dil
                high_var, otsu_t = U.otsu_wall(var_2d, comp_wall, U.OTSU_FACTOR)
                y0, y1, x0, x1 = comp["bbox"]
                y0, y1, x0, x1 = U.expand_bbox_square(
                    (y0, y1 - 1, x0, x1 - 1), H, W, 10, 50)
                side = "left" if (x0 + x1) / 2 < W / 2 else "right"
                title = f"ESUS {esus_id} -> nn{nn_id} | z={z} | {side} | Otsu={otsu_t:.0f}"
                U.draw_wall(cta[z], ct_win, lumen[z], plaque[z], wall_region[z],
                            est_outer[z], var_2d, high_var, otsu_t,
                            (y0, y1, x0, x1), title)
                
                
