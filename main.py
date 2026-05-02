"""
PIPELINE KLASIFIKASI DUPLIKASI LAPORAN
"""

import sys
import io

# Atur stdout ke UTF-8 agar aman di Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# XGBoost - install terpisah: pip install xgboost
from xgboost import XGBClassifier

# LightGBM - install terpisah: pip install lightgbm
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (aman untuk script)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Konfigurasi global
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Threshold heuristik untuk labeling duplikat
THRESHOLD_JARAK_KM = 0.05 
THRESHOLD_JAM = 2.0

# FUNGSI UTILITAS
def haversine_vectorized(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:

    R = 6371.0

    # Konversi derajat ke radian
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    # Formula Haversine
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def hitung_selisih_jam(
    waktu_1: pd.Series,
    waktu_2: pd.Series,
) -> pd.Series:
    delta = (waktu_2 - waktu_1).abs()
    return delta.dt.total_seconds() / 3600.0

# LOAD & PERSIAPAN DATA
def load_data(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    print(f"[OK] Dataset dimuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    return df

# Pairing Data
def generate_pairs(df, max_distance_km=0.2, max_time_hours=3):
    df = df.sort_values(["DESCRIPTION_GROUPED", "ADDDATE"])
    df = df.copy().reset_index(drop=True)
    df = df.sort_values("ADDDATE").reset_index(drop=True)
    pairs = []

    for i in range(len(df)):
        row_i = df.iloc[i]

        for j in range(i + 1, len(df)):
            row_j = df.iloc[j]

            time_diff = (row_j["ADDDATE"] - row_i["ADDDATE"]).total_seconds() / 3600
            if time_diff > max_time_hours:
                break
            # Filter kategori
            if row_i["DESCRIPTION_GROUPED"] != row_j["DESCRIPTION_GROUPED"]:
                continue
            # Hitung jarak
            dist = haversine_vectorized(
                pd.Series([row_i["LATITUDE"]]),
                pd.Series([row_i["LONGITUDE"]]),
                pd.Series([row_j["LATITUDE"]]),
                pd.Series([row_j["LONGITUDE"]]),
            ).values[0]
            if dist > max_distance_km:
                continue

            pairs.append({
                "lat_1": row_i["LATITUDE"],
                "lon_1": row_i["LONGITUDE"],
                "lat_2": row_j["LATITUDE"],
                "lon_2": row_j["LONGITUDE"],
                "waktu_1": row_i["ADDDATE"],
                "waktu_2": row_j["ADDDATE"],
                "kategori_1": row_i["DESCRIPTION_GROUPED"],
                "kategori_2": row_j["DESCRIPTION_GROUPED"],
            })

    df_pairs = pd.DataFrame(pairs)
    print(f"[OK] Total pairs terbentuk: {len(df_pairs):,}")
    return df_pairs

# FEATURE ENGINEERING
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # VALIDASI KOLOM WAJIB
    required_cols = [
        "lat_1", "lon_1",
        "lat_2", "lon_2",
        "waktu_1", "waktu_2",
        "kategori_1", "kategori_2"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    # CONVERT DATETIME
    for col in ["waktu_1", "waktu_2"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # FILTER KOORDINAT VALID (FIX UTAMA)
    before = len(df)

    df = df[
        df["lat_1"].between(-90, 90) &
        df["lat_2"].between(-90, 90) &
        df["lon_1"].between(-180, 180) &
        df["lon_2"].between(-180, 180)
    ]

    after = len(df)
    if before != after:
        print(f"[!!] Menghapus {before - after:,} baris (koordinat tidak valid)")

    # DROP MISSING CRITICAL
    before = len(df)
    df = df.dropna(subset=[
        "lat_1", "lon_1",
        "lat_2", "lon_2",
        "waktu_1", "waktu_2"
    ])

    after = len(df)
    if before != after:
        print(f"[!!] Menghapus {before - after:,} baris (missing critical)")
    # HITUNG JARAK (HAVERSINE)
    df["selisih_jarak"] = haversine_vectorized(
        df["lat_1"], df["lon_1"],
        df["lat_2"], df["lon_2"],
    )
    # HITUNG SELISIH WAKTU
    df["selisih_jam"] = hitung_selisih_jam(
        df["waktu_1"], df["waktu_2"]
    )
    # FLAG KATEGORI SAMA
    df["kategori_sama"] = (
        df["kategori_1"] == df["kategori_2"]
    ).astype(int)
    # LOG TRANSFORM
    df["log_jarak"] = np.log1p(df["selisih_jarak"])
    df["log_jam"] = np.log1p(df["selisih_jam"])
    # INTERACTION FEATURE
    df["interaction"] = df["selisih_jarak"] * df["selisih_jam"]
    # DROP OUTLIER EKSTREM
    df = df[
        (df["selisih_jarak"] < 100) &
        (df["selisih_jam"] < 168)
    ]
    print("[OK] Feature engineering selesai:")
    print(f"   - Jumlah data: {len(df):,}")
    print(f"   - selisih_jarak -- min: {df['selisih_jarak'].min():.4f} km | max: {df['selisih_jarak'].max():.2f} km")
    print(f"   - selisih_jam   -- min: {df['selisih_jam'].min():.2f} jam | max: {df['selisih_jam'].max():.2f} jam")
    print(f"   - kategori_sama -- {df['kategori_sama'].mean():.1%}")
    return df

# HEURISTIC LABELING
def heuristic_labeling(
    df: pd.DataFrame,
    threshold_jarak_km: float = THRESHOLD_JARAK_KM,
    threshold_jam: float = THRESHOLD_JAM,
) -> pd.DataFrame:
    
    df = df.copy()
    df["is_duplicate"] = (
        (df["selisih_jarak"] < threshold_jarak_km)
        & (df["selisih_jam"] < threshold_jam)
        & (df["kategori_sama"] == 1)
    ).astype(int)

    n_dup = df["is_duplicate"].sum()
    n_total = len(df)
    ratio = n_dup / n_total if n_total > 0 else 0

    print(f"\n[OK] Heuristic labeling selesai:")
    print(f"   - Duplikat     : {n_dup:,}  ({ratio:.1%})")
    print(f"   - Non-Duplikat : {n_total - n_dup:,}  ({1 - ratio:.1%})")
    print(f"   - Threshold    : jarak < {threshold_jarak_km * 1000:.0f} m, "
          f"waktu < {threshold_jam:.1f} jam, kategori harus sama")

    return df

# PERSIAPAN FITUR UNTUK MODELING
def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "is_duplicate",
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # PILIH FITUR
    if feature_cols is None:
        feature_cols = [
            "log_jarak",
            "log_jam",
            "interaction",
        ]
    # VALIDASI KOLOM
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom tidak ditemukan: {missing_cols}")
    # AMBIL FITUR & TARGET
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    # HANDLE INFINITE VALUE
    X = X.replace([np.inf, -np.inf], np.nan)
    # ENCODE KATEGORIKAL
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    # PASTIKAN NUMERIC
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # DROP DATA INVALID
    mask = X.notna().all(axis=1) & y.notna()
    if (~mask).any():
        n_drop = (~mask).sum()
        print(f"[!!] Menghapus {n_drop:,} baris (NaN / invalid).")
        X, y = X.loc[mask], y.loc[mask] 
    # INFO DISTRIBUSI TARGET
    pos = int(y.sum())
    total = len(y)
    neg = total - pos

    print(f"\n[OK] Fitur siap:")
    print(f"   - Shape X: {X.shape}")
    print(f"   - Shape y: {y.shape}")
    print(f"   - Duplikat (1): {pos:,}")
    print(f"   - Non-duplikat (0): {neg:,}")
    print(f"   - Ratio: {pos/total:.2%}")

    if total > 0 and (pos / total) < 0.05:
        print("[WARNING] Data sangat imbalanced (<5% duplikat)")

    if X.isna().any().any():
        raise ValueError("Masih ada NaN di fitur setelah cleaning!")

    return X, y

# TRAINING & EVALUASI
def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\n{'='*60}")
    print(f" MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Train : {X_train.shape[0]:,} sampel "
          f"(dup={y_train.sum():,}, non-dup={len(y_train)-y_train.sum():,})")
    print(f"  Test  : {X_test.shape[0]:,} sampel "
          f"(dup={y_test.sum():,}, non-dup={len(y_test)-y_test.sum():,})")

    # scale_pos_weight untuk handle imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Inisialisasi model
    if model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            use_label_encoder=False,
            verbosity=0,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "logistic_regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            )),
        ])
    elif model_name == "linear_svm":
        base_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            )),
        ])
        model = CalibratedClassifierCV(base_svm, cv=3)
    elif model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1,
        )
    else:
        raise ValueError(
            f"Model '{model_name}' tidak dikenali. "
            f"Pilih: 'xgboost', 'random_forest', 'logistic_regression', 'linear_svm', 'lightgbm'."
        )

    # Training
    print(f"\n>> Melatih {model_name}...")
    model.fit(X_train, y_train)
    print("[OK] Training selesai!")

    # Prediksi
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    # Evaluasi 
    print(f"\n{'-'*60}")
    print(" CLASSIFICATION REPORT")
    print(f"{'-'*60}")
    report = classification_report(
        y_test, y_pred,
        target_names=["Non-Duplikat (0)", "Duplikat (1)"],
    )
    print(report)

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"  >> ROC-AUC Score : {roc_auc:.4f}")

    # Confusion Matrix 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Duplikat", "Duplikat"],
    )
    disp.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(f"Confusion Matrix - {model_name.upper()}", fontsize=13, fontweight="bold")

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_label = "Feature Importance"
    elif hasattr(model, "named_steps"):
        clf_step = model.named_steps.get("clf", None)
        if clf_step is not None and hasattr(clf_step, "coef_"):
            importances = np.abs(clf_step.coef_[0])
        else:
            importances = np.zeroes(X.shape[1])  # fallback
        imp_label = "Coefficient (abs)"
    else:
        importances = np.ones(X.shape[1])
        imp_label = "N/A (Calibrated SVM)"

    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    feat_imp.plot.barh(ax=axes[1], color=sns.color_palette("viridis", len(feat_imp)))
    axes[1].set_title(f"{imp_label} - {model_name.upper()}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(imp_label)

    plt.tight_layout()
    plt.savefig(f"evaluation_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   >> Plot tersimpan: evaluation_{model_name}.png")

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "classification_report": report,
    }


# RINGKASAN PERBANDINGAN MODEL

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "ROC-AUC": res["roc_auc"],
        })
    comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    print(f"\n{'='*60}")
    print(" PERBANDINGAN MODEL")
    print(f"{'='*60}")
    print(comparison.to_string(index=False))
    return comparison

# MAIN PIPELINE
def main():
    print("=" * 60)
    print(" PIPELINE KLASIFIKASI DUPLIKASI LAPORAN")
    print("=" * 60)
    # LOAD DATA HASIL PREPROCESSING
    DATA_PATH = "dataset_2024_final.csv"

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    print(f"[OK] Dataset loaded: {df_raw.shape}")
    required_cols = ["LATITUDE", "LONGITUDE", "ADDDATE", "DESCRIPTION_GROUPED"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df_raw["ADDDATE"] = pd.to_datetime(df_raw["ADDDATE"], errors="coerce")

    df_raw = df_raw.dropna(subset=["LATITUDE", "LONGITUDE", "ADDDATE"])

    print(f"[OK] Setelah cleaning: {df_raw.shape}")

    # GENERATE PAIRS
    print("\n[STEP] Generate pairs...")
    df_pairs = generate_pairs(df_raw)

    if len(df_pairs) == 0:
        raise ValueError("Tidak ada pasangan terbentuk. Cek threshold pairing!")
    if len(df_pairs) < 100:
        print("[WARNING] Pair terlalu sedikit, model bisa tidak stabil")
    # FEATURE ENGINEERING
    df_pairs = feature_engineering(df_pairs)
    # HEURISTIC LABELING
    df_pairs = heuristic_labeling(
        df_pairs,
        threshold_jarak_km=THRESHOLD_JARAK_KM,
        threshold_jam=THRESHOLD_JAM,
    )
    # PREPARE FEATURES
    X, y = prepare_features(df_pairs)
    # TRAINING & EVALUASI
    results = {}

    models = [
        ("Logistic Regression", "logistic_regression"),
        ("Linear SVM", "linear_svm"),
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
    ]

    for name, model_key in models:
        results[name] = train_and_evaluate(X, y, model_name=model_key)
    # PERBANDINGAN MODEL
    comparison = compare_models(results)
    # SAVE OUTPUT
    output_path = "data_berlabel_pairs.csv"
    df_pairs.to_csv(output_path, index=False)

    print(f"\n>> Dataset pasangan berlabel tersimpan: {output_path}")

    print("\n" + "=" * 60)
    print(" PIPELINE SELESAI [OK]")
    print("=" * 60)

    return df_pairs, results, comparison

# ENTRY POINT
if __name__ == "__main__":
    df, results, comparison = main()
