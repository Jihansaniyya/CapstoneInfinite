"""
=============================================================================
PIPELINE KLASIFIKASI DUPLIKASI LAPORAN (311 Service Requests)
=============================================================================
Pipeline ini mencakup:
  1. Preprocessing & Feature Engineering (Haversine, selisih jam)
  2. Heuristic Labeling (rule-based is_duplicate)
  3. Modeling (XGBoost + Random Forest)
  4. Evaluasi (Classification Report, ROC-AUC, Confusion Matrix)

Author : Capstone Team – Infinite Learning Internship
Updated: 2026-05-01
=============================================================================
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

# ── Konfigurasi global ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Threshold heuristik untuk labeling duplikat
THRESHOLD_JARAK_KM = 0.05   # 50 meter  = 0.05 km
THRESHOLD_JAM = 2.0          # 2 jam


# =============================================================================
# BAGIAN 1 — FUNGSI UTILITAS
# =============================================================================

def haversine_vectorized(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Menghitung jarak Haversine (dalam kilometer) secara vectorized.

    Formula Haversine menghitung jarak great-circle antara dua titik
    di permukaan bumi berdasarkan koordinat latitude & longitude.

    Parameters
    ----------
    lat1, lon1 : pd.Series — Koordinat titik pertama (derajat).
    lat2, lon2 : pd.Series — Koordinat titik kedua (derajat).

    Returns
    -------
    pd.Series — Jarak dalam kilometer.
    """
    R = 6371.0  # Radius bumi dalam km

    # Konversi derajat → radian (vectorized, tanpa loop)
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
    """
    Menghitung selisih waktu absolut antara dua kolom timestamp
    dalam satuan jam (float).

    Parameters
    ----------
    waktu_1, waktu_2 : pd.Series — Kolom bertipe datetime64.

    Returns
    -------
    pd.Series — Selisih absolut dalam jam.
    """
    delta = (waktu_2 - waktu_1).abs()
    return delta.dt.total_seconds() / 3600.0


# =============================================================================
# BAGIAN 2 — LOAD & PERSIAPAN DATA
# =============================================================================

def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.
    Mendukung format .csv dan .xlsx.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    print(f"[OK] Dataset dimuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    return df


def buat_data_dummy(n: int = 5_000) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dummy untuk demonstrasi pipeline.

    Dataset mensimulasikan pasangan laporan dengan koordinat,
    timestamp, dan kategori — mencerminkan data DC 311 yang
    sudah di-pair.

    Parameters
    ----------
    n : int — Jumlah baris (pasangan laporan).

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Koordinat pusat Washington DC ≈ (38.9072, -77.0369)
    base_lat, base_lon = 38.9072, -77.0369

    # Kategori layanan (mirip DC 311 service codes)
    kategori_list = [
        "Pothole",
        "Streetlight",
        "Trash Collection",
        "Parking Violation",
        "Noise Complaint",
        "Graffiti",
        "Water Leak",
        "Sidewalk Repair",
    ]

    lat_1 = base_lat + rng.normal(0, 0.02, n)
    lon_1 = base_lon + rng.normal(0, 0.02, n)

    # ~40% pasangan sengaja dibuat "dekat" agar ada duplikat
    close_mask = rng.random(n) < 0.4
    lat_2 = np.where(close_mask, lat_1 + rng.normal(0, 0.0003, n), lat_1 + rng.normal(0, 0.01, n))
    lon_2 = np.where(close_mask, lon_1 + rng.normal(0, 0.0003, n), lon_1 + rng.normal(0, 0.01, n))

    # Timestamp
    start = pd.Timestamp("2024-01-01")
    waktu_1 = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="min")
    # Pasangan "dekat" → selisih waktu kecil; sisanya acak
    selisih_menit = np.where(close_mask, rng.integers(0, 90, n), rng.integers(0, 72 * 60, n))
    waktu_2 = waktu_1 + pd.to_timedelta(selisih_menit, unit="min")

    # Kategori — pasangan "dekat" → 80% kemungkinan sama
    kat_1 = rng.choice(kategori_list, n)
    kat_2 = np.where(
        close_mask & (rng.random(n) < 0.8),
        kat_1,
        rng.choice(kategori_list, n),
    )

    df = pd.DataFrame({
        "lat_1": lat_1,
        "lon_1": lon_1,
        "lat_2": lat_2,
        "lon_2": lon_2,
        "waktu_1": waktu_1,
        "waktu_2": waktu_2,
        "kategori_1": kat_1,
        "kategori_2": kat_2,
    })

    print(f"[OK] Data dummy dibuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    return df


# =============================================================================
# BAGIAN 3 — FEATURE ENGINEERING
# =============================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur-fitur turunan dari data pasangan laporan:
      - selisih_jarak (km)  → Haversine
      - selisih_jam (jam)    → delta waktu absolut
      - kategori_sama (bool) → apakah kategori cocok
    """
    df = df.copy()

    # --- 1. Pastikan kolom waktu bertipe datetime ---
    for col in ["waktu_1", "waktu_2"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- 2. Hitung selisih jarak (Haversine, km) ---
    df["selisih_jarak"] = haversine_vectorized(
        df["lat_1"], df["lon_1"],
        df["lat_2"], df["lon_2"],
    )

    # --- 3. Hitung selisih jam ---
    df["selisih_jam"] = hitung_selisih_jam(df["waktu_1"], df["waktu_2"])

    # --- 4. Flag kategori sama ---
    df["kategori_sama"] = (df["kategori_1"] == df["kategori_2"]).astype(int)

    print("[OK] Feature engineering selesai:")
    print(f"   - selisih_jarak -- min: {df['selisih_jarak'].min():.4f} km, "
          f"max: {df['selisih_jarak'].max():.2f} km, "
          f"median: {df['selisih_jarak'].median():.4f} km")
    print(f"   - selisih_jam   -- min: {df['selisih_jam'].min():.2f} jam, "
          f"max: {df['selisih_jam'].max():.2f} jam")
    print(f"   - kategori_sama -- {df['kategori_sama'].sum():,} dari {len(df):,} "
          f"({df['kategori_sama'].mean():.1%})")

    return df


# =============================================================================
# BAGIAN 4 — HEURISTIC LABELING (RULE-BASED)
# =============================================================================

def heuristic_labeling(
    df: pd.DataFrame,
    threshold_jarak_km: float = THRESHOLD_JARAK_KM,
    threshold_jam: float = THRESHOLD_JAM,
) -> pd.DataFrame:
    """
    Membuat label target `is_duplicate` secara heuristik (rule-based).

    Aturan — sebuah pasangan dianggap DUPLIKAT (1) jika **semua**
    kondisi berikut terpenuhi:
      1. selisih_jarak  < threshold_jarak_km  (default: 0.05 km = 50 m)
      2. selisih_jam    < threshold_jam        (default: 2.0 jam)
      3. kategori_sama == 1

    Parameters
    ----------
    df : pd.DataFrame — DataFrame yang sudah punya fitur turunan.
    threshold_jarak_km : float — Batas jarak (km).
    threshold_jam : float — Batas waktu (jam).

    Returns
    -------
    pd.DataFrame — Dengan kolom baru `is_duplicate`.
    """
    df = df.copy()

    # Semua kondisi harus terpenuhi (AND logis) ─ vectorized
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


# =============================================================================
# BAGIAN 5 — PERSIAPAN FITUR UNTUK MODELING
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "is_duplicate",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Menyiapkan matriks fitur (X) dan vektor target (y).

    Fitur default:
      - selisih_jarak
      - selisih_jam
      - kategori_sama

    Kolom kategorikal tambahan akan di-encode dengan LabelEncoder.
    """
    if feature_cols is None:
        feature_cols = ["selisih_jarak", "selisih_jam", "kategori_sama"]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode kolom object/kategorikal jika ada
    le_dict = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Hapus baris dengan NaN di fitur
    mask = X.notna().all(axis=1)
    if (~mask).any():
        n_drop = (~mask).sum()
        print(f"[!!] Menghapus {n_drop:,} baris dengan NaN di fitur.")
        X, y = X.loc[mask], y.loc[mask]

    print(f"\n[OK] Fitur siap: X{X.shape}, y{y.shape}")
    return X, y


# =============================================================================
# BAGIAN 6 — TRAINING & EVALUASI
# =============================================================================

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Pipeline pelatihan dan evaluasi model.

    Parameters
    ----------
    X : pd.DataFrame — Matriks fitur.
    y : pd.Series — Vektor target (0/1).
    model_name : str — 'xgboost' atau 'random_forest'.
    test_size : float — Proporsi data test.
    random_state : int — Random seed untuk reproduktibilitas.

    Returns
    -------
    dict — Berisi model, y_test, y_pred, y_proba, dan metrik.
    """

    # ── 1. Train-Test Split (stratified untuk menjaga rasio kelas) ──
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

    # ── 2. Hitung scale_pos_weight untuk handle imbalanced data ──
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # ── 3. Inisialisasi model ──
    if model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,  # Mengatasi imbalance
            eval_metric="logloss",
            random_state=random_state,
            use_label_encoder=False,
            verbosity=0,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",  # Mengatasi imbalance
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "logistic_regression":
        # Pipeline dengan StandardScaler karena LR sensitif terhadap skala fitur
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",  # Mengatasi imbalance
                max_iter=1000,
                random_state=random_state,
            )),
        ])
    elif model_name == "linear_svm":
        # LinearSVC tidak punya predict_proba, dibungkus CalibratedClassifierCV
        # Pipeline dengan StandardScaler karena SVM sensitif terhadap skala fitur
        base_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                class_weight="balanced",  # Mengatasi imbalance
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
            scale_pos_weight=scale_pos_weight,  # Mengatasi imbalance
            random_state=random_state,
            verbose=-1,
        )
    else:
        raise ValueError(
            f"Model '{model_name}' tidak dikenali. "
            f"Pilih: 'xgboost', 'random_forest', 'logistic_regression', 'linear_svm', 'lightgbm'."
        )

    # ── 4. Training ──
    print(f"\n>> Melatih {model_name}...")
    model.fit(X_train, y_train)
    print("[OK] Training selesai!")

    # ── 5. Prediksi ──
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── 6. Evaluasi ──
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

    # ── 7. Confusion Matrix Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Duplikat", "Duplikat"],
    )
    disp.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(f"Confusion Matrix - {model_name.upper()}", fontsize=13, fontweight="bold")

    # Feature Importance / Koefisien
    if hasattr(model, "feature_importances_"):
        # Tree-based models (XGBoost, RF, LightGBM)
        importances = model.feature_importances_
        imp_label = "Feature Importance"
    elif hasattr(model, "named_steps"):
        # Pipeline (Logistic Regression) — ambil koefisien absolut
        clf_step = model.named_steps.get("clf", None)
        if clf_step is not None and hasattr(clf_step, "coef_"):
            importances = np.abs(clf_step.coef_[0])
        else:
            importances = np.ones(X.shape[1])  # fallback
        imp_label = "Coefficient (abs)"
    else:
        # CalibratedClassifierCV (Linear SVM) — tidak ada feature importance
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


# =============================================================================
# BAGIAN 7 — RINGKASAN PERBANDINGAN MODEL
# =============================================================================

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Membuat tabel perbandingan ROC-AUC dari beberapa model.
    """
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


# =============================================================================
# BAGIAN 8 — MAIN PIPELINE
# =============================================================================

def main():
    """
    Menjalankan seluruh pipeline end-to-end:
      1. Load / generate data
      2. Feature engineering
      3. Heuristic labeling
      4. Persiapan fitur
      5. Training & evaluasi (XGBoost + Random Forest)
      6. Perbandingan model
    """

    print("=" * 60)
    print(" PIPELINE KLASIFIKASI DUPLIKASI LAPORAN")
    print("=" * 60)

    # ── STEP 1: Load data ──
    # Ganti path di bawah ini dengan file dataset Anda yang sesungguhnya.
    # Contoh: df = load_data("data/Xfinal.csv")
    #
    # Jika dataset belum tersedia, gunakan data dummy untuk demo:
    DATA_PATH = None  # ← Ubah ke path file CSV/XLSX Anda

    if DATA_PATH and Path(DATA_PATH).exists():
        df = load_data(DATA_PATH)
        # ─────────────────────────────────────────────────────────
        # PENTING: Sesuaikan nama kolom di bawah ini dengan dataset
        # Anda yang sebenarnya. Mapping contoh:
        #
        # df = df.rename(columns={
        #     "LATITUDE": "lat_1",
        #     "LONGITUDE": "lon_1",
        #     "LATITUDE_2": "lat_2",
        #     "LONGITUDE_2": "lon_2",
        #     "ADDDATE": "waktu_1",
        #     "RESOLUTIONDATE": "waktu_2",
        #     "SERVICECODE": "kategori_1",
        #     "SERVICECODE_2": "kategori_2",
        # })
        # ─────────────────────────────────────────────────────────
    else:
        print("[!!] DATA_PATH belum diatur atau file tidak ditemukan.")
        print("     Menggunakan data dummy untuk demonstrasi.\n")
        df = buat_data_dummy(n=5_000)

    # ── STEP 2: Feature Engineering ──
    df = feature_engineering(df)

    # ── STEP 3: Heuristic Labeling ──
    df = heuristic_labeling(
        df,
        threshold_jarak_km=THRESHOLD_JARAK_KM,
        threshold_jam=THRESHOLD_JAM,
    )

    # ── STEP 4: Persiapan Fitur ──
    X, y = prepare_features(df)

    # ── STEP 5: Training & Evaluasi ──
    results = {}

    # 5a. Logistic Regression
    results["Logistic Regression"] = train_and_evaluate(X, y, model_name="logistic_regression")

    # 5b. Linear SVM
    results["Linear SVM"] = train_and_evaluate(X, y, model_name="linear_svm")

    # 5c. Random Forest
    results["Random Forest"] = train_and_evaluate(X, y, model_name="random_forest")

    # 5d. XGBoost
    results["XGBoost"] = train_and_evaluate(X, y, model_name="xgboost")

    # 5e. LightGBM
    results["LightGBM"] = train_and_evaluate(X, y, model_name="lightgbm")

    # ── STEP 6: Perbandingan Model ──
    comparison = compare_models(results)

    # ── STEP 7: Simpan dataset berlabel ──
    output_path = "data_berlabel.csv"
    df.to_csv(output_path, index=False)
    print(f"\n>> Dataset berlabel tersimpan: {output_path}")

    print(f"\n{'='*60}")
    print(" PIPELINE SELESAI [OK]")
    print(f"{'='*60}")

    return df, results, comparison


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    df, results, comparison = main()
