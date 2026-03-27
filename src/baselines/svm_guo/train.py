# Reproduction of Guo et al. (2024):
#   "Feasibility of ultrasound radiomics based models for classification
#    of liver fibrosis due to Schistosoma japonicum infection"
#   PLoS Negl Trop Dis 18(6): e0012235
#
# Original method summary:
#   - Feature extraction: Pyradiomics (first-order, GLCM, GLDM, GLRLM,
#     GLSZM + wavelet)
#   - Feature selection: (1) Mann-Whitney U test, (2) LASSO
#   - Classifier: SVM with RBF kernel
#   - Oversampling: SMOTE for class imbalance
#   - Hyperparameter tuning: 5-fold stratified cross-validation
#   - Evaluation: 6 pairwise binary classification groups
#
# Adaptation notes:
#   The original paper uses 6 pairwise groups (Grade 0-1, 0-2, 0-3, 1-2,
#   1-3, 2-3). We additionally include 3 composite groups (0 vs 1-3,
#   1 vs 2-3, 1-2 vs 3) for more comprehensive comparison in our study.
#
# Reproducibility note:
#   The original source code was not publicly accessible when this study was
#   conducted. This implementation is an independent re-implementation written
#   for fair comparison and released with the SFibAI repository.
#
# Runtime (rough): 9 pairs × (feature extraction + U-test + LASSO + 5-fold
# GridSearch SVM). With cache: ~2–10 min per pair (SVM dominant). Without
# cache: Pyradiomics ~0.5–2 min/sample → e.g. 1000 train+val ~8–30 h once,
# then cache makes later runs fast. Total 9 pairs with cache often 30–90 min.
#
# Device: Entire pipeline runs on CPU (sklearn / Pyradiomics have no GPU impl).
# Parallelism: GridSearchCV/LassoCV use n_jobs within one process; run multiple
# processes with different --pairs for parallel runs. Use --max_train_samples
# for subsampling and --no_cv to skip 5-fold grid search on large datasets.
#
# Example usage (command line):
#
#   # Default: 9 pairs, 5-fold GridSearch, paths from config if available
#   python train.py
#
#   # Custom data and output paths
#   python train.py --data_root /path/to/cls_raw4 --runs_dir /path/to/runs
#
#   # Large dataset: subsample 2000 train samples per pair, skip 5-fold CV (faster)
#   python train.py --no_cv --max_train_samples 2000 --n_jobs 16
#   
#   # Run only a subset of pairs (e.g. first 3 pairwise)
#   python train.py --pairs 0-1 0-2 0-3
#
#   # Parallel runs: start 3 processes, each with different --pairs
#   python train.py --pairs 0-1 0-2 0-3 --no_cv --max_train_samples 10800 --max_val_samples 1200 &
#   python train.py --pairs 1-2 1-3 2-3 --no_cv --max_train_samples 10800 --max_val_samples 1200 &
#   python train.py --pairs "0-1,2,3" "1-2,3" "1,2-3" --no_cv --max_train_samples 10800 --max_val_samples 1200 &
#
#   # Disable wavelet features and clear feature cache
#   python train.py --no_wavelet --clear_cache
#
#   # Fewer CV folds and custom feature cache dir
#   python train.py --cv_folds 3 --features_cache_dir /path/to/features

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_image(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def bbox_txt_to_mask(img_shape: Tuple[int, int],
                     yolo_bbox: List[float]) -> np.ndarray:
    h, w = img_shape
    x_c, y_c, bw, bh = yolo_bbox
    x_c *= w; y_c *= h; bw *= w; bh *= h
    left = int(max(0, x_c - bw / 2))
    top = int(max(0, y_c - bh / 2))
    right = int(min(w, x_c + bw / 2))
    bottom = int(min(h, y_c + bh / 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def load_split_samples(root: Path, split: str) -> List[Tuple[str, int]]:
    """Load ``(image_path, label)`` pairs from ``<root>/<split>/<grade>/``."""
    samples: List[Tuple[str, int]] = []
    split_dir = root / split
    if not split_dir.exists():
        return samples
    for cls_name in ["0", "1", "2", "3"]:
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            samples.append((str(cls_dir / fname), int(cls_name)))
    return samples


def match_label_txt(image_path: str,
                    split_label_dir: Path) -> Optional[str]:
    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    p = split_label_dir / base
    return str(p) if p.exists() else None


def read_yolo_bbox(txt_path: str) -> Optional[List[float]]:
    try:
        with open(txt_path, "r") as f:
            line = f.readline().strip()
            if not line:
                return None
            parts = line.split()
            if len(parts) != 5:
                return None
            return list(map(float, parts[1:]))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Radiomics feature extraction
# ---------------------------------------------------------------------------

def get_radiomics_extractor(enable_wavelet: bool = True):
    try:
        from radiomics import featureextractor
    except ImportError as e:
        raise RuntimeError(
            "pyradiomics is required. Install it with pip, for example: "
            "pip install pyradiomics==3.1.0"
        ) from e

    params = {
        "imageType": {"Original": {}},
        "featureClass": {
            "firstorder": [],
            "glcm": [],
            "gldm": [],
            "glrlm": [],
            "glszm": [],
        },
        "setting": {
            "normalize": True,
            "binWidth": 25,
            "resampledPixelSpacing": None,
            "force2D": True,
            "correctMask": True,
        },
    }
    if enable_wavelet:
        params["imageType"]["Wavelet"] = {}

    return featureextractor.RadiomicsFeatureExtractor(params)


def extract_features_for_sample(extractor, image: np.ndarray,
                                mask: np.ndarray) -> Dict[str, float]:
    import SimpleITK as sitk
    image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    result = extractor.execute(image_sitk, mask_sitk)
    return {k: float(v) for k, v in result.items()
            if isinstance(v, (int, float, np.floating))}


def get_extractor_config_hash(extractor, enable_wavelet: bool) -> str:
    try:
        conf = {
            "imageType": getattr(extractor, "imageType", {}),
            "enabledFeatures": getattr(extractor, "enabledFeatures", {}),
            "settings": getattr(extractor, "settings", {}),
            "enable_wavelet": enable_wavelet,
        }
    except Exception:
        conf = {"repr": str(extractor), "enable_wavelet": enable_wavelet}
    conf_str = json.dumps(conf, sort_keys=True, default=str)
    return hashlib.md5(conf_str.encode()).hexdigest()[:10]

# ---------------------------------------------------------------------------
# Feature selection (following Guo et al.: Mann-Whitney U -> LASSO)
# ---------------------------------------------------------------------------

def u_test_feature_selection(df: pd.DataFrame, labels: np.ndarray,
                             alpha: float = 0.05) -> List[str]:
    """Step 1: Mann-Whitney U test to remove non-discriminative features."""
    kept = []
    for col in df.columns:
        try:
            g0 = df[col][labels == 0].values
            g1 = df[col][labels == 1].values
            if len(g0) < 2 or len(g1) < 2:
                continue
            _, p = mannwhitneyu(g0, g1, alternative='two-sided')
            if p < alpha:
                kept.append(col)
        except Exception:
            continue
    return kept


def lasso_feature_selection(df: pd.DataFrame, labels: np.ndarray,
                            n_folds: int = 5, seed: int = 42) -> List[str]:
    """Step 2: LASSO regression for sparse feature selection."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    model = LassoCV(cv=n_folds, random_state=seed, max_iter=5000, n_jobs=-1)
    model.fit(X, labels)
    coef = np.abs(model.coef_)
    selected_idx = np.where(coef > 1e-8)[0]
    return [df.columns[i] for i in selected_idx]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_pair_subset(samples: List[Tuple[str, int]],
                      pair: Tuple[List[int], List[int]]
                      ) -> List[Tuple[str, int]]:
    """Build binary-classification subset.  group1 -> label 0, group2 -> 1."""
    group1, group2 = pair
    result = []
    for p, y in samples:
        if y in group1:
            result.append((p, 0))
        elif y in group2:
            result.append((p, 1))
    return result


def stratified_sample(X: pd.DataFrame, y: np.ndarray, max_samples: int,
                     seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """Stratified subsample to at most max_samples. Returns (X_sub, y_sub)."""
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx].copy(), y[idx]


def stratified_sample_pairs(samples: List[Tuple[str, int]], max_samples: int,
                            seed: int = 42) -> List[Tuple[str, int]]:
    """Stratified subsample (path, label) pairs to at most max_samples (before feature extraction)."""
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    y = np.array([label for _, label in samples])
    X_dummy = pd.DataFrame(np.arange(len(y)))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    idx, _ = next(sss.split(X_dummy, y))
    return [samples[i] for i in idx]


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    if len(aucs) == 0:
        return float('nan'), float('nan')
    lower = np.percentile(aucs, 100 * (alpha / 2))
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def evaluate_metrics(y_true, y_prob,
                     calc_auc_ci: bool = True) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    auc = roc_auc_score(y_true, y_prob)
    auc_ci = (None, None)
    if calc_auc_ci:
        auc_ci = bootstrap_auc_ci(y_true, y_prob)
    return {
        "auc": auc,
        "auc_ci_lower": auc_ci[0], "auc_ci_upper": auc_ci[1],
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sens, "specificity": spec,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

# ---------------------------------------------------------------------------
# Feature extraction with disk caching
# ---------------------------------------------------------------------------

def _extract_features_with_cache(args, extractor, samples_pair,
                                 split_label_dir, split_name,
                                 features_cache_dir):
    """Extract radiomics features for *samples_pair*, using disk cache."""
    features_cache_dir.mkdir(parents=True, exist_ok=True)
    config_hash = get_extractor_config_hash(extractor, not args.no_wavelet)
    cache_file = features_cache_dir / f"{split_name}_features_{config_hash}.pkl"

    print(f"\nExtracting {split_name} features ...")

    # Try loading from cache
    cached_df = None
    if args.use_cache and cache_file.exists():
        try:
            cached_df = pd.read_pickle(cache_file)
            print(f"  Loaded {len(cached_df)} cached records from {cache_file}")
        except Exception as e:
            print(f"  Failed to load cache: {e}")
            cached_df = None

    current_paths = {p for p, _ in samples_pair}

    if cached_df is not None:
        cached_paths = set(cached_df['image_path'].tolist())
        missing_paths = current_paths - cached_paths
    else:
        missing_paths = current_paths

    # Extract missing features
    if missing_paths:
        samples_to_extract = [(p, y) for p, y in samples_pair
                              if p in missing_paths]
        print(f"  Extracting {len(samples_to_extract)} samples ...")
        new_feats, new_paths = [], []
        for i, (img_path, _) in enumerate(samples_to_extract):
            if (i + 1) % 50 == 0 or i == len(samples_to_extract) - 1:
                print(f"    Progress: {i+1}/{len(samples_to_extract)} "
                      f"({len(new_feats)} success)")
            img = read_image(img_path)
            if img is None:
                continue
            txt = match_label_txt(img_path, split_label_dir)
            if txt is None:
                continue
            bbox = read_yolo_bbox(txt)
            if bbox is None:
                continue
            mask = bbox_txt_to_mask(img.shape, bbox)
            try:
                feats = extract_features_for_sample(extractor, img, mask)
                new_feats.append(feats)
                new_paths.append(img_path)
            except Exception as e:
                if len(new_feats) < 5:
                    print(f"      Failed: {os.path.basename(img_path)} – {e}")
                continue

        if new_feats:
            new_df = (pd.DataFrame(new_feats)
                      .replace([np.inf, -np.inf], np.nan).fillna(0.0))
            new_df['image_path'] = new_paths
            if cached_df is not None:
                cached_df = pd.concat([cached_df, new_df], ignore_index=True)
            else:
                cached_df = new_df
            try:
                cached_df.to_pickle(cache_file)
                print(f"  Cache updated -> {cache_file}")
            except Exception as e:
                print(f"  Cache save failed: {e}")
        elif cached_df is None:
            return pd.DataFrame(), np.array([])
    else:
        print(f"  All {len(samples_pair)} samples found in cache")

    # Filter to current samples and align labels
    filtered = cached_df[cached_df['image_path'].isin(current_paths)].copy()
    path2label = {p: y for p, y in samples_pair}
    labels = np.array([path2label[p] for p in filtered['image_path'].tolist()])
    feats_df = filtered.drop(columns=['image_path', 'label'], errors='ignore')

    print(f"  {split_name}: {len(feats_df)} samples, "
          f"{feats_df.shape[1]} features")
    return feats_df, labels

# ---------------------------------------------------------------------------
# Train & evaluate one binary pair
# ---------------------------------------------------------------------------

def run_one_pair(args, extractor, train_root: Path, val_root: Path,
                 pair: Tuple[List[int], List[int]], save_dir: Path,
                 features_cache_dir: Path):
    g1_str = "-".join(map(str, sorted(pair[0])))
    g2_str = "-".join(map(str, sorted(pair[1])))
    pair_name = f"Grade_{g1_str}_vs_Grade_{g2_str}"
    out_dir = save_dir / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training: {pair_name}")
    print(f"  Group 1 (label 0): Grade {g1_str}")
    print(f"  Group 2 (label 1): Grade {g2_str}")
    print(f"{'=' * 60}")

    # Load & build binary subsets, then optionally stratify-sample before feature extraction
    train_samples = load_split_samples(train_root.parent, "train")
    val_samples = load_split_samples(val_root.parent, "val")
    train_pair = build_pair_subset(train_samples, pair)
    val_pair = build_pair_subset(val_samples, pair)
    if getattr(args, "max_train_samples", 0) > 0:
        n_before = len(train_pair)
        train_pair = stratified_sample_pairs(train_pair, args.max_train_samples, seed=args.seed)
        if len(train_pair) < n_before:
            print(f"  Train sampled (before feature extraction): {n_before} -> {len(train_pair)} (--max_train_samples={args.max_train_samples})")
    if getattr(args, "max_val_samples", 0) > 0:
        n_before = len(val_pair)
        val_pair = stratified_sample_pairs(val_pair, args.max_val_samples, seed=args.seed)
        if len(val_pair) < n_before:
            print(f"  Val sampled (before feature extraction): {n_before} -> {len(val_pair)} (--max_val_samples={args.max_val_samples})")

    n0_tr = sum(1 for _, y in train_pair if y == 0)
    n1_tr = sum(1 for _, y in train_pair if y == 1)
    n0_va = sum(1 for _, y in val_pair if y == 0)
    n1_va = sum(1 for _, y in val_pair if y == 1)
    print(f"  Train: {len(train_pair)} (G1={n0_tr}, G2={n1_tr})")
    print(f"  Val:   {len(val_pair)} (G1={n0_va}, G2={n1_va})")

    train_label_dir = train_root.parent / "train_label"
    val_label_dir = val_root.parent / "val_label"

    # Feature extraction
    X_train_df, y_train = _extract_features_with_cache(
        args, extractor, train_pair, train_label_dir, "training",
        features_cache_dir)
    X_val_df, y_val = _extract_features_with_cache(
        args, extractor, val_pair, val_label_dir, "validation",
        features_cache_dir)

    if X_train_df.empty or X_val_df.empty:
        print("  ERROR: empty features, skipping this pair")
        return

    # Align feature columns
    cols = sorted(set(X_train_df.columns) & set(X_val_df.columns))
    X_train_df = X_train_df[cols]
    X_val_df = X_val_df[cols]
    print(f"\nAligned features: {len(cols)}")

    # ---- Feature selection (Guo et al.: U-test -> LASSO) ------------------
    # Step 1: Mann-Whitney U test
    print(f"\nStep 1: Mann-Whitney U test (alpha={args.u_test_alpha}) ...")
    kept_u = u_test_feature_selection(X_train_df, y_train,
                                      alpha=args.u_test_alpha)
    print(f"  Kept {len(kept_u)} / {len(cols)} features")
    if len(kept_u) == 0:
        kept_u = cols[:min(100, len(cols))]
        print(f"  U-test kept nothing, falling back to first {len(kept_u)}")

    X_train_u = X_train_df[kept_u]

    # Step 2: LASSO
    print(f"Step 2: LASSO feature selection ...")
    kept_lasso = lasso_feature_selection(X_train_u, y_train, seed=args.seed)
    print(f"  Kept {len(kept_lasso)} / {len(kept_u)} features")
    if len(kept_lasso) == 0:
        kept_lasso = kept_u[:min(50, len(kept_u))]
        print(f"  LASSO kept nothing, falling back to first {len(kept_lasso)}")

    X_train = X_train_df[kept_lasso]
    X_val = X_val_df[kept_lasso]

    # ---- Train subsampling (fallback when sampling was not done before extraction) ----
    if getattr(args, "max_train_samples", 0) > 0 and len(y_train) > args.max_train_samples:
        n_before = len(y_train)
        X_train, y_train = stratified_sample(
            X_train, y_train, args.max_train_samples, seed=args.seed)
        print(f"\n  Train subsampled: {n_before} -> {len(y_train)} (--max_train_samples={args.max_train_samples})")

    # ---- SVM: 5-fold GridSearch or single fit (--no_cv for speed) ----------
    min_class_count = min(np.bincount(y_train))
    k_neighbors = max(1, min(5, min_class_count - 1))

    pipe = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel='rbf', probability=True, class_weight='balanced')),
    ])

    if getattr(args, "no_cv", False):
        # Skip 5-fold; single fit with fixed C=1, gamma='scale'
        print(f"\nTraining SVM (RBF) with fixed C=1, gamma='scale' (--no_cv, no GridSearch) ...")
        pipe.set_params(svc__C=1, svc__gamma='scale')
        start_time = time.time()
        pipe.fit(X_train, y_train)
        training_time = time.time() - start_time
        best_model = pipe
        best_params = {"svc__C": 1, "svc__gamma": "scale"}
        best_cv_auc = float("nan")
    else:
        n_cv = min(args.cv_folds, min_class_count)
        param_grid = {
            "svc__C": [0.01, 0.1, 1, 10, 100],
            "svc__gamma": ['scale', 'auto', 0.001, 0.01, 0.1],
        }
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipe, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=args.n_jobs, refit=True, verbose=0)
        print(f"\nTraining SVM (RBF) with {args.cv_folds}-fold CV GridSearch ...")
        start_time = time.time()
        grid.fit(X_train, y_train)
        training_time = time.time() - start_time
        best_params = grid.best_params_
        best_model = grid.best_estimator_
        best_cv_auc = float(grid.best_score_)
        print(f"  Best params: {best_params}")
        print(f"  Best CV AUC: {best_cv_auc:.4f}")
    print(f"  Training time: {training_time:.1f}s")

    # ---- Evaluate ---------------------------------------------------------
    print(f"\nEvaluating ...")
    y_prob_train = best_model.predict_proba(X_train)[:, 1]
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    metrics_train = evaluate_metrics(y_train, y_prob_train, calc_auc_ci=True)
    metrics_val = evaluate_metrics(y_val, y_prob_val, calc_auc_ci=True)

    # Save model
    try:
        import joblib
        joblib.dump({
            "model": best_model,
            "features": kept_lasso,
            "best_params": best_params,
        }, out_dir / "model.joblib")
    except Exception:
        pass

    report = {
        "pair": pair,
        "pair_name": pair_name,
        "best_params": best_params,
        "best_cv_auc": best_cv_auc,
        "training_time": training_time,
        "train_metrics": metrics_train,
        "val_metrics": metrics_val,
        "num_features_after_utest": len(kept_u),
        "num_features_after_lasso": len(kept_lasso),
        "features": kept_lasso,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Completed: {pair_name}")
    print(f"  Params: {best_params}")
    print(f"  Features: {len(cols)} -> U-test {len(kept_u)} -> LASSO "
          f"{len(kept_lasso)}")
    _print_metrics("Train", metrics_train)
    _print_metrics("Val  ", metrics_val)
    print(f"{'=' * 60}")

    # Append to summary CSV
    _append_summary_csv(save_dir, pair_name, metrics_train, metrics_val)


def _print_metrics(prefix: str, m: Dict):
    ci_lo = m.get('auc_ci_lower', float('nan'))
    ci_hi = m.get('auc_ci_upper', float('nan'))
    print(f"  {prefix}: AUC={m['auc']:.3f} "
          f"[{ci_lo:.3f}-{ci_hi:.3f}]  "
          f"Acc={m['acc']:.3f}  F1={m['f1']:.3f}  "
          f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}")


def _append_summary_csv(save_dir, pair_name, m_tr, m_va):
    try:
        rows = []
        for split, m in [("train", m_tr), ("validation", m_va)]:
            rows.append({
                "pair": pair_name, "split": split,
                "auc": m["auc"],
                "auc_ci": f"[{m['auc_ci_lower']:.3f}, {m['auc_ci_upper']:.3f}]",
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "accuracy": m["acc"],
                "precision": m["precision"],
                "f1": m["f1"],
                "kappa": m["kappa"],
            })
        summary_path = save_dir / "summary.csv"
        df_add = pd.DataFrame(rows)
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            df_add = pd.concat([old, df_add], ignore_index=True)
        df_add.to_csv(summary_path, index=False)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_paths():
    """Build default paths relative to the current repository root."""
    repo_root = Path(__file__).resolve().parents[3]
    data_root = repo_root / "data" / "baseline_data" / "cls_raw4"
    runs_root = repo_root / "artifacts" / "runs"
    cache_root = repo_root / "artifacts" / "feature_cache" / "svm_guo"
    return str(data_root), str(runs_root), str(cache_root)


def parse_args():
    _dp = _default_paths()
    p = argparse.ArgumentParser(
        description="Reproduction of Guo et al. Radiomics + SVM for liver "
                    "fibrosis classification")
    p.add_argument("--data_root", type=str, default=_dp[0],
                   help="Data root with train/0,1,2,3/, val/, train_label, val_label")
    p.add_argument("--runs_dir", type=str, default=_dp[1],
                   help="Root dir for runs (creates guosvm_YYYYMMDD_HHMMSS under it)")
    p.add_argument("--features_cache_dir", type=str, default=_dp[2],
                   help="Pyradiomics feature cache directory")

    # Classification pairs
    # NOTE: The first 6 pairs correspond to the original Guo et al. paper.
    #       The last 3 composite pairs are extensions for our comparative study.
    p.add_argument("--pairs", nargs="*", type=str,
                   default=[
                       # Original 6 pairwise groups (Guo et al. Table 2)
                       "0-1", "0-2", "0-3", "1-2", "1-3", "2-3",
                       # Extended composite groups (for our paper Table 1)
                       "0-1,2,3", "1-2,3", "1,2-3",
                   ])

    # Feature selection
    p.add_argument("--u_test_alpha", type=float, default=0.05,
                   help="Mann-Whitney U test significance level")

    # SVM and training scale
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Number of CV folds for GridSearchCV (ignored if --no_cv)")
    p.add_argument("--no_cv", action="store_true",
                   help="Skip 5-fold grid search; single fit with C=1, gamma='scale' (faster on large data)")
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="Max training samples per binary pair (0=no limit). Stratified sampling before feature extraction.")
    p.add_argument("--max_val_samples", type=int, default=0,
                   help="Max validation samples per binary pair (0=no limit). Stratified sampling before feature extraction.")
    p.add_argument("--n_jobs", type=int, default=8,
                   help="Number of CPU jobs for GridSearchCV/LassoCV")
    p.add_argument("--seed", type=int, default=42)

    # Feature extraction
    p.add_argument("--no_wavelet", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--use_cache", action="store_true", default=True)
    p.add_argument("--clear_cache", action="store_true")

    return p.parse_args()


def _parse_pair_string(s: str) -> Tuple[List[int], List[int]]:
    """Parse pair spec: '0-1' -> ([0],[1]), '0-1,2,3' -> ([0],[1,2,3])."""
    if "-" not in s:
        raise ValueError(f"Invalid pair format: {s}")
    left, right = s.split("-", 1)
    g1 = [int(x.strip()) for x in left.split(",")]
    g2 = [int(x.strip()) for x in right.split(",")]
    return g1, g2


def main():
    args = parse_args()

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    features_cache_dir = Path(args.features_cache_dir)
    if args.clear_cache and features_cache_dir.exists():
        import shutil
        shutil.rmtree(features_cache_dir)
        print("Feature cache cleared")

    data_root = Path(args.data_root)
    runs_dir = Path(args.runs_dir) / f"guosvm_{time.strftime('%Y%m%d_%H%M%S')}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    extractor = get_radiomics_extractor(enable_wavelet=not args.no_wavelet)
    train_root = data_root / "train"
    val_root = data_root / "val"

    pairs: List[Tuple[List[int], List[int]]] = []
    for s in args.pairs:
        try:
            g1, g2 = _parse_pair_string(s)
            if set(g1) & set(g2):
                print(f"Warning: overlapping pair {s}, skipped")
                continue
            pairs.append((g1, g2))
        except Exception as e:
            print(f"Warning: invalid pair {s}: {e}")

    print(f"\n{'#' * 60}")
    print(f"Guo et al. Radiomics + SVM Reproduction")
    print(f"  {len(pairs)} binary tasks (default 9: 6 pairwise + 3 composite)")
    print(f"  Feature selection: Mann-Whitney U -> LASSO")
    cv_desc = "fixed C/gamma (--no_cv)" if getattr(args, "no_cv", False) else f"{args.cv_folds}-fold CV"
    _mt = getattr(args, "max_train_samples", 0)
    _mv = getattr(args, "max_val_samples", 0)
    samp_parts = []
    if _mt > 0:
        samp_parts.append(f"max_train_samples={_mt}")
    if _mv > 0:
        samp_parts.append(f"max_val_samples={_mv}")
    samp_desc = ", " + ", ".join(samp_parts) if samp_parts else ""
    print(f"  Model: SVM (RBF) + SMOTE, {cv_desc}{samp_desc}")
    print(f"  Results -> {runs_dir}")
    print(f"{'#' * 60}")

    for i, pair in enumerate(pairs, 1):
        g1s = "-".join(map(str, sorted(pair[0])))
        g2s = "-".join(map(str, sorted(pair[1])))
        origin_tag = "" if len(pair[0]) == 1 and len(pair[1]) == 1 \
            else " [extended]"
        print(f"  {i}. Grade {g1s} vs Grade {g2s}{origin_tag}")

    for i, pair in enumerate(pairs, 1):
        print(f"\n{'#' * 60}")
        print(f"Progress: {i}/{len(pairs)}")
        print(f"{'#' * 60}")
        run_one_pair(args, extractor, train_root, val_root, pair,
                     runs_dir, features_cache_dir)

    print(f"\n{'#' * 60}")
    print(f"All training completed!")
    print(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results: {runs_dir}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
