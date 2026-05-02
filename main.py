"""
=============================================================================
PIPELINE BINARY CLASSIFICATION — DETEKSI DUPLIKASI LAPORAN LAYANAN PUBLIK
=============================================================================
Pipeline 6 langkah:
  1. Problem Framing & Algorithm Selection
  2. Data Preparation (Pair Generation, Haversine, Labeling, Anti-Leakage)
  3. Model Initialization & Training
  4. Model Evaluation
  5. Hyperparameter Tuning
  6. Model Saving

Dataset : dataset_2024_final.csv (DC 311 Service Requests 2024)
Target  : is_duplicate (0 = Non-Duplikat, 1 = Duplikat)

Author : Capstone Team — Infinite Learning Internship
Updated: 2026-05-02
=============================================================================
"""

import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from math import radians, cos, sin, asin, sqrt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Konfigurasi Global ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = "dataset_2024_final.csv"
MODEL_DIR = Path("saved_models")

# Threshold labeling
DIST_THRESHOLD_KM = 0.1    # < 0.1 km (100 meter)
TIME_THRESHOLD_HR = 4.0    # < 4 jam
NOISE_STD_DIST = 0.02      # Gaussian noise std untuk dist_delta
NOISE_STD_TIME = 0.5       # Gaussian noise std untuk time_delta
LABEL_FLIP_RATE = 0.015    # Flip 1.5% label


# =============================================================================
# HAVERSINE FORMULA
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Hitung jarak Haversine antara dua titik koordinat (dalam KM)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# =============================================================================
# LANGKAH 1 — PROBLEM FRAMING & ALGORITHM SELECTION
# =============================================================================

def step1_problem_framing():
    """Langkah 1: Problem Framing & Algorithm Selection."""
    print("=" * 70)
    print(" LANGKAH 1: PROBLEM FRAMING & ALGORITHM SELECTION")
    print("=" * 70)
    print("""
  Tipe Masalah   : Binary Classification
  Target         : is_duplicate (0=Non-Duplikat, 1=Duplikat)
  Features       : dist_delta, time_delta, HOUR, DAYOFWEEK
  Label Logic    : Duplikat jika dist < 0.1km AND time < 4jam AND kategori sama
  Anti-Leakage   : Gaussian noise + label flipping + remove smoking gun

  Algoritma      :
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. LightGBM
    5. Linear SVM
    """)

    models = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "Linear SVM": "linear_svm",
    }
    return models


# =============================================================================
# LANGKAH 2 — DATA PREPARATION
# =============================================================================

def step2_data_preparation(filepath: str) -> tuple:
    """
    Langkah 2: Data Preparation.
    - Load CSV, parse datetime
    - Generate candidate pairs (blocking by category + time window)
    - Compute features: Haversine distance, time delta
    - Heuristic labeling
    - Anti-leakage: noise + label flip
    - Split train/test
    """
    print("=" * 70)
    print(" LANGKAH 2: DATA PREPARATION")
    print("=" * 70)

    # --- 2a. Load & parse ---
    df = pd.read_csv(filepath)
    print(f"\n  [OK] Dataset dimuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")

    df["ADDDATE"] = pd.to_datetime(df["ADDDATE"], utc=True)
    needed = ["LATITUDE", "LONGITUDE", "ADDDATE", "SERVICECODEDESCRIPTION",
              "HOUR", "DAYOFWEEK"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    df = df.dropna(subset=needed)
    print(f"  [OK] Data bersih: {len(df):,} baris")

    # --- 2b. Pair Generation (Vectorized Blocking Strategy) ---
    print("\n  >> Generating pairs (vectorized blocking)...")

    np.random.seed(RANDOM_STATE)

    categories = df["SERVICECODEDESCRIPTION"].unique()
    print(f"  [OK] {len(categories)} kategori unik ditemukan")

    # --- Same-category pairs (vectorized: shift-based pairing) ---
    MAX_PAIRS_PER_CAT = 5000
    same_pair_frames = []

    for cat in categories:
        cat_df = (df[df["SERVICECODEDESCRIPTION"] == cat]
                  .sort_values("ADDDATE")
                  .reset_index(drop=True))
        if len(cat_df) < 2:
            continue

        # Vectorized: pair row[i] with row[i+offset] for offset 1..5
        cat_pairs = []
        for offset in range(1, 6):
            if offset >= len(cat_df):
                break
            a = cat_df.iloc[:-offset].reset_index(drop=True)
            b = cat_df.iloc[offset:].reset_index(drop=True)
            td_hours = (b["ADDDATE"].values - a["ADDDATE"].values).astype("timedelta64[s]").astype(float) / 3600
            mask = td_hours <= 24
            if mask.sum() == 0:
                continue
            chunk = pd.DataFrame({
                "lat1": a.loc[mask, "LATITUDE"].values,
                "lon1": a.loc[mask, "LONGITUDE"].values,
                "lat2": b.loc[mask, "LATITUDE"].values,
                "lon2": b.loc[mask, "LONGITUDE"].values,
                "time1": a.loc[mask, "ADDDATE"].values,
                "time2": b.loc[mask, "ADDDATE"].values,
                "hour1": a.loc[mask, "HOUR"].values,
                "hour2": b.loc[mask, "HOUR"].values,
                "dow1": a.loc[mask, "DAYOFWEEK"].values,
                "dow2": b.loc[mask, "DAYOFWEEK"].values,
                "same_category": 1,
            })
            cat_pairs.append(chunk)

        if cat_pairs:
            merged = pd.concat(cat_pairs, ignore_index=True)
            if len(merged) > MAX_PAIRS_PER_CAT:
                merged = merged.sample(MAX_PAIRS_PER_CAT, random_state=RANDOM_STATE)
            same_pair_frames.append(merged)

    same_pairs = pd.concat(same_pair_frames, ignore_index=True) if same_pair_frames else pd.DataFrame()
    n_same = len(same_pairs)
    print(f"  [OK] Same-category pairs: {n_same:,}")

    # --- Different-category pairs (vectorized random sampling) ---
    n_diff_target = min(n_same, 50000)
    idx1 = np.random.choice(len(df), size=n_diff_target * 2, replace=True)
    idx2 = np.random.choice(len(df), size=n_diff_target * 2, replace=True)
    r1 = df.iloc[idx1].reset_index(drop=True)
    r2 = df.iloc[idx2].reset_index(drop=True)
    diff_mask = r1["SERVICECODEDESCRIPTION"].values != r2["SERVICECODEDESCRIPTION"].values
    r1 = r1[diff_mask].iloc[:n_diff_target].reset_index(drop=True)
    r2 = r2[diff_mask].iloc[:n_diff_target].reset_index(drop=True)

    diff_pairs = pd.DataFrame({
        "lat1": r1["LATITUDE"].values, "lon1": r1["LONGITUDE"].values,
        "lat2": r2["LATITUDE"].values, "lon2": r2["LONGITUDE"].values,
        "time1": r1["ADDDATE"].values, "time2": r2["ADDDATE"].values,
        "hour1": r1["HOUR"].values, "hour2": r2["HOUR"].values,
        "dow1": r1["DAYOFWEEK"].values, "dow2": r2["DAYOFWEEK"].values,
        "same_category": 0,
    })

    print(f"  [OK] Different-category pairs: {len(diff_pairs):,}")

    pdf = pd.concat([same_pairs, diff_pairs], ignore_index=True)
    print(f"  [OK] Total pairs: {len(pdf):,}")

    # --- 2c. Feature Engineering ---
    print("\n  >> Computing features...")

    # Raw features (untuk labeling)
    pdf["dist_delta_raw"] = haversine(
        pdf["lat1"], pdf["lon1"], pdf["lat2"], pdf["lon2"]
    )
    pdf["time_delta_raw"] = abs(
        (pdf["time2"] - pdf["time1"]).dt.total_seconds() / 3600
    )

    # --- 2d. Heuristic Labeling ---
    pdf["is_duplicate"] = (
        (pdf["dist_delta_raw"] < DIST_THRESHOLD_KM) &
        (pdf["time_delta_raw"] < TIME_THRESHOLD_HR) &
        (pdf["same_category"] == 1)
    ).astype(int)

    n_dup = pdf["is_duplicate"].sum()
    n_non = len(pdf) - n_dup
    print(f"\n  [OK] Label distribution (sebelum flip):")
    print(f"       Duplikat     : {n_dup:,} ({n_dup/len(pdf)*100:.1f}%)")
    print(f"       Non-Duplikat : {n_non:,} ({n_non/len(pdf)*100:.1f}%)")

    # --- 2e. Anti-Leakage ---
    print("\n  >> Applying anti-leakage measures...")

    # 1) Gaussian noise pada fitur numerik
    pdf["dist_delta"] = pdf["dist_delta_raw"] + np.random.normal(0, NOISE_STD_DIST, len(pdf))
    pdf["dist_delta"] = pdf["dist_delta"].clip(lower=0)
    pdf["time_delta"] = pdf["time_delta_raw"] + np.random.normal(0, NOISE_STD_TIME, len(pdf))
    pdf["time_delta"] = pdf["time_delta"].clip(lower=0)
    print(f"  [OK] Gaussian noise ditambahkan (dist_std={NOISE_STD_DIST}, time_std={NOISE_STD_TIME})")

    # 2) Random label flipping (1.5%)
    n_flip = int(len(pdf) * LABEL_FLIP_RATE)
    flip_idx = np.random.choice(pdf.index, size=n_flip, replace=False)
    pdf.loc[flip_idx, "is_duplicate"] = 1 - pdf.loc[flip_idx, "is_duplicate"]
    print(f"  [OK] Label flip: {n_flip:,} labels ({LABEL_FLIP_RATE*100:.1f}%)")

    # 3) Remove smoking gun: same_category TIDAK jadi fitur training
    print("  [OK] Smoking gun 'same_category' dihapus dari fitur training")

    n_dup_final = pdf["is_duplicate"].sum()
    n_non_final = len(pdf) - n_dup_final
    print(f"\n  [OK] Label distribution (setelah flip):")
    print(f"       Duplikat     : {n_dup_final:,} ({n_dup_final/len(pdf)*100:.1f}%)")
    print(f"       Non-Duplikat : {n_non_final:,} ({n_non_final/len(pdf)*100:.1f}%)")

    # --- 2f. Prepare features & Split ---
    # Fitur: dist_delta (noised), time_delta (noised), HOUR (avg), DAYOFWEEK (avg)
    pdf["HOUR"] = ((pdf["hour1"] + pdf["hour2"]) / 2).astype(float)
    pdf["DAYOFWEEK"] = ((pdf["dow1"] + pdf["dow2"]) / 2).astype(float)

    feature_cols = ["dist_delta", "time_delta", "HOUR", "DAYOFWEEK"]
    X = pdf[feature_cols]
    y = pdf["is_duplicate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    print(f"\n  [OK] Split data (stratified):")
    print(f"       Train : {len(X_train):,} ({y_train.sum():,} dup)")
    print(f"       Test  : {len(X_test):,} ({y_test.sum():,} dup)\n")

    return X_train, X_test, y_train, y_test, feature_cols


# =============================================================================
# LANGKAH 3 — MODEL INITIALIZATION & TRAINING
# =============================================================================

def step3_train_model(X_train, y_train, model_name: str):
    """Langkah 3: Model Initialization & Training."""
    print(f"\n  >> Inisialisasi & Training: {model_name.upper()}")

    if model_name == "logistic_regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ])
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, n_jobs=-1,
            scale_pos_weight=max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        )
    elif model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
            is_unbalance=True,
        )
    elif model_name == "linear_svm":
        base_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                max_iter=2000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ])
        model = CalibratedClassifierCV(base_svm, cv=3)
    else:
        raise ValueError(f"Model '{model_name}' tidak dikenali.")

    model.fit(X_train, y_train)
    print(f"  [OK] Training {model_name} selesai!")
    return model


# =============================================================================
# LANGKAH 4 — MODEL EVALUATION
# =============================================================================

def step4_evaluate(model, X_train, y_train, X_test, y_test, model_name: str):
    """
    Langkah 4: Model Evaluation.
    Metrik: Accuracy, F1, ROC-AUC, PR-AUC, CV mean/std, Gap.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    pr_auc_val = average_precision_score(y_test, y_proba) if y_proba is not None else None

    # Cross-validation (5-fold)
    print(f"  >> Running 5-fold Cross-Validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = abs(cv_mean - roc) if roc is not None else None

    print(f"\n  {'─'*55}")
    print(f"  EVALUASI: {model_name.upper()}")
    print(f"  {'─'*55}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  ROC-AUC (test) : {roc:.4f}" if roc else "  ROC-AUC (test) : N/A")
    print(f"  PR-AUC  (test) : {pr_auc_val:.4f}" if pr_auc_val else "  PR-AUC  (test) : N/A")
    print(f"  CV mean        : {cv_mean:.4f}")
    print(f"  CV std         : {cv_std:.4f}")
    if gap is not None:
        print(f"  Gap            : {gap:.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=["Non-Duplikat", "Duplikat"],
        zero_division=0,
    )
    print(f"\n{report}")

    # --- Plots: Confusion Matrix + ROC + Precision-Recall ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1) Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non-Dup", "Dup"])
    disp.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(f"Confusion Matrix\n{model_name.upper()}", fontweight="bold")

    if y_proba is not None:
        # 2) ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(f"ROC Curve\n{model_name.upper()}", fontweight="bold")
        axes[1].legend()

        # 3) Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[2].plot(rec, prec, "r-", lw=2, label=f"AP = {pr_auc_val:.3f}")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title(f"Precision-Recall Curve\n{model_name.upper()}", fontweight="bold")
        axes[2].legend()
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"evaluation_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Plot tersimpan: evaluation_{model_name}.png")

    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "pr_auc": pr_auc_val,
            "cv_mean": cv_mean, "cv_std": cv_std, "gap": gap}


# =============================================================================
# LANGKAH 5 — HYPERPARAMETER TUNING (XGBoost)
# =============================================================================

def step5_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Langkah 5: Hyperparameter Tuning — RandomizedSearchCV pada XGBoost."""
    print("\n" + "=" * 70)
    print(" LANGKAH 5: HYPERPARAMETER TUNING (XGBoost)")
    print("=" * 70)

    scale_pw = max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1,
        scale_pos_weight=scale_pw,
    )

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        model, param_distributions=param_dist,
        n_iter=20, cv=5, scoring="f1",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )

    print("  >> Menjalankan RandomizedSearchCV (5-fold CV)...")
    search.fit(X_train, y_train)

    print(f"\n  [OK] Best Parameters: {search.best_params_}")
    print(f"  [OK] Best CV F1: {search.best_score_:.4f}")

    tuned = search.best_estimator_
    y_pred = tuned.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    y_proba = tuned.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n  [OK] Performa setelah tuning (test set):")
    print(f"       Accuracy  : {acc:.4f}")
    print(f"       F1-Score  : {f1:.4f}")
    print(f"       ROC-AUC   : {roc:.4f}\n")

    report = classification_report(
        y_test, y_pred,
        target_names=["Non-Duplikat", "Duplikat"],
        zero_division=0,
    )
    print(report)

    return tuned, search.best_params_


# =============================================================================
# LANGKAH 6 — MODEL SAVING
# =============================================================================

def step6_save_model(model, model_name: str, feature_cols: list):
    """Langkah 6: Model Saving — Simpan model & metadata dengan joblib."""
    print("\n" + "=" * 70)
    print(" LANGKAH 6: MODEL SAVING")
    print("=" * 70)

    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / f"best_model_{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"  [OK] Model tersimpan  : {model_path}")

    metadata = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "target": "is_duplicate",
        "task": "binary_classification",
        "labels": {0: "Non-Duplikat", 1: "Duplikat"},
        "thresholds": {
            "dist_km": DIST_THRESHOLD_KM,
            "time_hr": TIME_THRESHOLD_HR,
        },
    }
    meta_path = MODEL_DIR / "metadata.pkl"
    joblib.dump(metadata, meta_path)
    print(f"  [OK] Metadata         : {meta_path}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Menjalankan seluruh pipeline end-to-end (6 langkah)."""

    print("\n" + "=" * 70)
    print(" PIPELINE DETEKSI DUPLIKASI LAPORAN LAYANAN PUBLIK DC 311")
    print("=" * 70 + "\n")

    # ── LANGKAH 1 ──
    model_registry = step1_problem_framing()

    # ── LANGKAH 2 ──
    X_train, X_test, y_train, y_test, feature_cols = \
        step2_data_preparation(DATA_PATH)

    # ── LANGKAH 3 & 4 ──
    print("=" * 70)
    print(" LANGKAH 3 & 4: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    results = {}
    trained_models = {}

    for display_name, model_key in model_registry.items():
        model = step3_train_model(X_train, y_train, model_key)
        metrics = step4_evaluate(model, X_train, y_train, X_test, y_test, model_key)
        results[display_name] = metrics
        trained_models[display_name] = (model_key, model)

    # ── Ringkasan Perbandingan ──
    print("\n" + "=" * 70)
    print(" RINGKASAN PERBANDINGAN MODEL")
    print("=" * 70)

    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "ROC-AUC (test)": m["roc_auc"],
            "PR-AUC (test)": m["pr_auc"],
            "CV mean": m["cv_mean"],
            "CV std": m["cv_std"],
            "Gap": m["gap"],
        })

    comparison = pd.DataFrame(rows).sort_values("ROC-AUC (test)", ascending=False)
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["Model"]
    best_key, best_model = trained_models[best_name]
    print(f"\n  >> Model terbaik: {best_name} (ROC-AUC={comparison.iloc[0]['ROC-AUC (test)']:.4f})")

    # ── LANGKAH 5 ──
    tuned_model, best_params = step5_hyperparameter_tuning(
        X_train, y_train, X_test, y_test
    )

    final_model = tuned_model
    final_name = "xgboost_tuned"

    # ── LANGKAH 6 ──
    step6_save_model(final_model, final_name, feature_cols)

    print("=" * 70)
    print(" PIPELINE SELESAI [OK]")
    print("=" * 70)

    return results, comparison


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results, comparison = main()
