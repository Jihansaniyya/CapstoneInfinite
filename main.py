# PIPELINE MODEL 1
import sys, warnings, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
from pathlib import Path
 
import joblib
 
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Threshold label
LABEL_KM  = 0.05
LABEL_JAM = 2.0 

# Window pairing
PAIRING_KM  = 0.10
PAIRING_JAM = 4.0

FEATURE_NAMES = [
    "hour_i",
    "hour_j",
    "dow_i",
    "month_i",
    "daystoclose_i",
    "daystoclose_j",
    "dtc_diff",
    "kat_enc",
]
# LOAD & VALIDASI
def load_and_validate(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    print("LOAD & VALIDASI DATASET")
    print("\n" + "=" * 65)
    t0 = time.time()

    df = (pd.read_excel(filepath)
          if filepath.suffix in (".xlsx", ".xls")
          else pd.read_csv(filepath))

    print(f"File    : {filepath.name}")
    print(f"Ukuran  : {df.shape[0]:,} baris x {df.shape[1]} kolom")

    REQUIRED = ["LATITUDE", "LONGITUDE", "ADDDATE", "DESCRIPTION_GROUPED"]
    missing  = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    df["ADDDATE"]   = pd.to_datetime(df["ADDDATE"], errors="coerce")
    df["LATITUDE"]  = pd.to_numeric(df["LATITUDE"],  errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    mask = (
        df["LATITUDE"].between(-90, 90)
        & df["LONGITUDE"].between(-180, 180)
        & df["ADDDATE"].notna()
        & df["DESCRIPTION_GROUPED"].notna()
    )
    n_drop = (~mask).sum()
    df     = df[mask].reset_index(drop=True)
    if n_drop:
        print(f"  Drop    : {n_drop:,} baris invalid")

    if "DAYSTOCLOSE" not in df.columns:
        print("  [INFO] Kolom DAYSTOCLOSE tidak ada — diisi 0")
        df["DAYSTOCLOSE"] = 0.0
    else:
        med = df.groupby("DESCRIPTION_GROUPED")["DAYSTOCLOSE"].transform("median")
        df["DAYSTOCLOSE"] = df["DAYSTOCLOSE"].fillna(med).fillna(0.0)

    print(f"  Valid   : {df.shape[0]:,} baris  [{time.time()-t0:.1f}s]")
    print(f"  Periode : {df['ADDDATE'].min().date()} s.d. "
          f"{df['ADDDATE'].max().date()}")
    print(f"\n  Distribusi kategori:")
    for kat, cnt in df["DESCRIPTION_GROUPED"].value_counts().items():
        print(f"    {kat:<45} {cnt:>7,}  ({cnt/len(df):.1%})")

    return df


# SPLIT DATA
def split_at_report_level(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("SPLIT DATA")
    print("=" * 65)
    print("  [PENTING] Split dilakukan SEBELUM generate pairs")
    print("  agar tidak ada laporan yang sama di train & test set")

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["DESCRIPTION_GROUPED"],
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    print(f"\n  Train: {len(df_train):,} laporan ({len(df_train)/len(df):.0%})")
    print(f"  Test : {len(df_test):,} laporan ({len(df_test)/len(df):.0%})")
    return df_train, df_test

# GENERATE PAIRS + FEATURES + LABELS
def generate_pairs_features_labels(
    df: pd.DataFrame,
    label_enc: LabelEncoder,
    pairing_km:  float = PAIRING_KM,
    pairing_jam: float = PAIRING_JAM,
    label_km:    float = LABEL_KM,
    label_jam:   float = LABEL_JAM,
    set_name:    str   = "train",
) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n  Generate pairs [{set_name}] "
          f"radius={pairing_km*1000:.0f}m, window={pairing_jam:.0f}j ...")
    t0 = time.time()

    df = df.copy()

    # Encode kategori
    df["_kat_enc"] = df["DESCRIPTION_GROUPED"].map(
        {cls: i for i, cls in enumerate(label_enc.classes_)}
    ).fillna(-1).astype(int)

    X_rows: list = []
    y_rows: list = []

    for kat_val in df["_kat_enc"].unique():
        if kat_val == -1:
            continue

        dk = (
            df[df["_kat_enc"] == kat_val]
            .sort_values("ADDDATE")
            .reset_index(drop=True)
        )
        if len(dk) < 2:
            continue

        # BallTree Haversine
        coords_rad = np.radians(dk[["LATITUDE", "LONGITUDE"]].values)
        nn = NearestNeighbors(
            radius=pairing_km / 6371.0,
            metric="haversine",
            algorithm="ball_tree",
        )
        nn.fit(coords_rad)
        dists_rad, nbrs = nn.radius_neighbors(coords_rad, return_distance=True)

        times_ns = dk["ADDDATE"].values.astype(np.int64)
        hour     = dk["ADDDATE"].dt.hour.values.astype(np.float32)
        dow      = dk["ADDDATE"].dt.dayofweek.values.astype(np.float32)
        month    = dk["ADDDATE"].dt.month.values.astype(np.float32)
        dtc      = dk["DAYSTOCLOSE"].values.astype(np.float32)

        for i in range(len(dk)):
            for k_idx in range(len(nbrs[i])):
                j = int(nbrs[i][k_idx])
                if j <= i:
                    continue

                dt_sec = float(times_ns[j] - times_ns[i]) / 1e9
                if dt_sec > pairing_jam * 3600:
                    break

                dist_km = float(dists_rad[i][k_idx]) * 6371.0
                dt_jam  = dt_sec / 3600.0

                label = int(dist_km < label_km and dt_jam < label_jam)

                X_rows.append([
                    hour[i],
                    hour[j],
                    dow[i],
                    month[i],
                    dtc[i],
                    dtc[j],
                    float(abs(dtc[i] - dtc[j])),
                    float(kat_val),
                ])
                y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)

    n_dup   = int(y.sum())
    n_total = len(y)
    ratio   = n_dup / n_total if n_total > 0 else 0

    print(f"    Pairs       : {n_total:,}  [{time.time()-t0:.1f}s]")
    print(f"    Duplikat    : {n_dup:,}  ({ratio:.1%})")
    print(f"    Non-dup     : {n_total - n_dup:,}  ({1-ratio:.1%})")
    return X, y


# EDA DATA TRAIN

def plot_eda(X: np.ndarray, y: np.ndarray,
             save_path: str = "eda_fitur.png"):
    print("EDA\n")

    df_p = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_p["label"] = y

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes  = axes.flatten()
    CDUP  = "#D85A30"
    CNDUP = "#185FA5"

    for i, col in enumerate(FEATURE_NAMES):
        ax = axes[i]
        for cls, color, lbl in [
            (0, CNDUP, f"Non-Dup (n={(y==0).sum():,})"),
            (1, CDUP,  f"Dup     (n={(y==1).sum():,})"),
        ]:
            data = df_p[df_p["label"] == cls][col].values
            ax.hist(data, bins=40, alpha=0.6, density=True,
                    color=color, label=lbl)
        ax.set_title(col, fontweight="bold", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "EDA: Distribusi Fitur per Kelas (Train Set)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# BASELINE LIGHTGBM
def train_baseline_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[LGBMClassifier, dict]:
    print("\n BASELINE LIGHTGBM")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    t_fit = time.time() - t0

    metrics = _evaluate(model, X_test, y_test, "LightGBM Baseline",
                        X_train, y_train, random_state)
    metrics["fit_time"] = t_fit
    print(f"  Waktu training : {t_fit:.1f}s")

    return model, metrics


# HYPERPARAMETER TUNING (RandomizedSearchCV)
def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_iter:       int = 30,
    cv_folds:     int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[LGBMClassifier, dict, dict]:
    print("\n HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print(f"  n_iter     : {n_iter}")
    print(f"  CV folds   : {cv_folds}")
    print(f"  Scoring    : roc_auc")
    print(f"  Total fits : {n_iter * cv_folds} (+ refit final)")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    base_model = LGBMClassifier(
        scale_pos_weight=scale,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )

    param_dist = {
        "num_leaves":         randint(20, 120),
        "max_depth":          randint(3, 10),
        "learning_rate":      uniform(0.01, 0.19),   # [0.01, 0.20]
        "n_estimators":       randint(100, 500),
        "subsample":          uniform(0.6, 0.4),     # [0.6, 1.0]
        "colsample_bytree":   uniform(0.6, 0.4),     # [0.6, 1.0]
        "min_child_samples":  randint(10, 100),
        "reg_alpha":          uniform(0.0, 1.0),
        "reg_lambda":         uniform(0.0, 5.0),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True, 
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )

    print(f"\n  Mulai tuning...")
    t0 = time.time()
    search.fit(X_train, y_train)
    t_tune = time.time() - t0

    best_params = search.best_params_
    best_model  = search.best_estimator_

    print(f"\n  Selesai [{t_tune:.0f}s]")
    print(f"  Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"\n  Best parameters:")
    for k, v in sorted(best_params.items()):
        v_fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<25} : {v_fmt}")

    # Evaluasi best model pada test set
    metrics = _evaluate(best_model, X_test, y_test, "LightGBM Tuned",
                        X_train, y_train, random_state)
    metrics["tune_time"]      = t_tune
    metrics["best_cv_score"]  = float(search.best_score_)
    metrics["best_params"]    = best_params

    # Simpan hasil tuning
    df_cv = pd.DataFrame(search.cv_results_)
    df_cv = df_cv.sort_values("rank_test_score")[
        ["rank_test_score", "mean_test_score", "std_test_score",
         "params", "mean_fit_time"]
    ].head(10)
    print(f"\n  Top-5 kombinasi parameter:")
    print(df_cv.head(5).to_string(index=False))

    return best_model, best_params, metrics


# EVALUASI 
def _evaluate(
    model,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    name:    str,
    X_full:  np.ndarray,
    y_full:  np.ndarray,
    random_state: int = RANDOM_STATE,
) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report  = classification_report(
        y_test, y_pred,
        target_names=["Non-Duplikat", "Duplikat"],
        digits=4,
    )
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_s = cross_val_score(
        model, X_full, y_full, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    print(f"\n {name}")
    print(report)
    print(f"  ROC-AUC (test)  : {roc_auc:.4f}")
    print(f"  PR-AUC  (test)  : {pr_auc:.4f}")
    print(f"  5-CV (train)    : {cv_s.mean():.4f} ± {cv_s.std():.4f}")

    gap = abs(roc_auc - cv_s.mean())
    if   gap < 0.02: print(f"  Generalisasi    : [OK] Sangat baik (gap={gap:.4f})")
    elif gap < 0.05: print(f"  Generalisasi    : [OK] Baik (gap={gap:.4f})")
    else:            print(f"  Generalisasi    : [!] Gap besar (gap={gap:.4f})")

    _plot_eval_single(model, X_test, y_test, y_pred, y_proba,
                      roc_auc, pr_auc, name)

    return {
        "model":   model,
        "name":    name,
        "roc_auc": roc_auc,
        "pr_auc":  pr_auc,
        "cv_mean": float(cv_s.mean()),
        "cv_std":  float(cv_s.std()),
        "report":  report,
    }


def _plot_eval_single(
    model, X_test, y_test, y_pred, y_proba,
    roc_auc, pr_auc, name: str,
):
    safe_name = name.lower().replace(" ", "_")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["Non-Dup", "Dup"],
    ).plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
    axes[0].set_title(f"Confusion Matrix\n{name}", fontweight="bold")

    # PR Curve 
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(rec, prec, "#185FA5", linewidth=2)
    axes[1].fill_between(rec, prec, alpha=0.12, color="#185FA5")
    baseline = float(y_test.mean())
    axes[1].axhline(baseline, color="gray", linestyle="--",
                    linewidth=1, label=f"Baseline ({baseline:.2f})")
    axes[1].set(
        xlabel="Recall", ylabel="Precision",
        title=f"Precision-Recall Curve — PR-AUC={pr_auc:.4f}\n{name}",
        xlim=[0, 1], ylim=[0, 1.05],
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].title.set_fontweight("bold")

    #  Feature Importance 
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pd.Series(imp, index=FEATURE_NAMES).sort_values().plot.barh(
            ax=axes[2],
            color=sns.color_palette("viridis", len(FEATURE_NAMES)),
        )
        axes[2].set_title(f"Feature Importance\n{name}", fontweight="bold")
        axes[2].grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"{name}  |  ROC-AUC={roc_auc:.4f}  |  PR-AUC={pr_auc:.4f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = f"eval_{safe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Plot            : {out}")


# PERBANDINGAN BASELINE vs TUNED
def compare_baseline_vs_tuned(
    baseline_metrics: dict,
    tuned_metrics:    dict,
    save_path: str = "perbandingan_tuning.png",
) -> pd.DataFrame:
    print(" PERBANDINGAN BASELINE vs TUNED")

    rows = [baseline_metrics, tuned_metrics]
    cols = ["name", "roc_auc", "pr_auc", "cv_mean", "cv_std"]
    df = pd.DataFrame([{c: r[c] for c in cols} for r in rows])
    df.columns = ["Model", "ROC-AUC (test)", "PR-AUC (test)",
                  "CV mean", "CV std"]
    df["CV gap"] = abs(df["ROC-AUC (test)"] - df["CV mean"])

    df_disp = df.copy()
    for col in df_disp.columns[1:]:
        df_disp[col] = df_disp[col].map("{:.4f}".format)
    print(df_disp.to_string(index=False))

    delta_roc = tuned_metrics["roc_auc"] - baseline_metrics["roc_auc"]
    delta_pr  = tuned_metrics["pr_auc"]  - baseline_metrics["pr_auc"]
    print(f"\n  Delta ROC-AUC  : {delta_roc:+.4f}")
    print(f"  Delta PR-AUC   : {delta_pr:+.4f}")
    if abs(delta_roc) < 0.005:
        print("  [INFO] Perbedaan < 0.005 — baseline sudah cukup optimal")

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_plot = ["ROC-AUC (test)", "PR-AUC (test)", "CV mean"]
    x = np.arange(len(metrics_plot))
    w = 0.35

    vals_base  = [baseline_metrics["roc_auc"],
                  baseline_metrics["pr_auc"],
                  baseline_metrics["cv_mean"]]
    vals_tuned = [tuned_metrics["roc_auc"],
                  tuned_metrics["pr_auc"],
                  tuned_metrics["cv_mean"]]

    b1 = ax.bar(x - w/2, vals_base,  w, label="Baseline", color="#185FA5", alpha=0.8)
    b2 = ax.bar(x + w/2, vals_tuned, w, label="Tuned",    color="#D85A30", alpha=0.8)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=8)

    ax.set(xticks=x, xticklabels=metrics_plot,
           ylim=[0, 1.1], ylabel="Score",
           title="LightGBM: Baseline vs Tuned")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  [OK] Plot perbandingan: {save_path}")

    return df


# MODEL SAVING


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


def save_model(
    model,
    best_params: dict,
    metrics:     dict,
    output_dir:  str = ".",
) -> str:
    import json
    print("MODEL SAVING")

    output_dir = Path(output_dir).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        fallback = Path.home() / "lightgbm_output"
        print(f"  [WARN] Tidak bisa tulis ke {output_dir}: {e}")
        print(f"  [INFO] Fallback ke: {fallback}")
        output_dir = fallback
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output dir : {output_dir}")

    #  Simpan model (.joblib)
    model_path = output_dir / "lightgbm_best.joblib"
    try:
        joblib.dump(model, model_path, compress=3)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Model     : {model_path}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  [ERROR] Gagal simpan model: {e}")
        raise

    if not best_params:
        print("  [INFO] best_params kosong → ambil dari model.get_params()")
        best_params = model.get_params()

    metadata_raw = {
        "model_type":      type(model).__name__,
        "roc_auc_test":    metrics.get("roc_auc", None),
        "pr_auc_test":     metrics.get("pr_auc",  None),
        "cv_mean":         metrics.get("cv_mean", None),
        "cv_std":          metrics.get("cv_std",  None),
        "feature_names":   FEATURE_NAMES,
        "n_features":      len(FEATURE_NAMES),
        "label_threshold": {
            "jarak_km": LABEL_KM,
            "jam":      LABEL_JAM,
        },
        "pairing_config": {
            "radius_km": PAIRING_KM,
            "jam":       PAIRING_JAM,
        },
        "best_params": best_params,
        "saved_at":    pd.Timestamp.now().isoformat(),
    }
    metadata = _make_json_serializable(metadata_raw)

    meta_path = output_dir / "lightgbm_metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        size_kb = meta_path.stat().st_size / 1024
        print(f"  [OK] Metadata  : {meta_path}  ({size_kb:.1f} KB)")
    except TypeError as e:
        print(f"  [WARN] JSON dump error: {e}")
        print(f"  [INFO] Coba dengan default=str ...")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [OK] Metadata  : {meta_path} (dengan fallback default=str)")
    except Exception as e:
        print(f"  [ERROR] Gagal simpan metadata: {e}")
        raise

    print("\n  Verifikasi: mencoba load model kembali...")
    try:
        model_loaded = joblib.load(model_path)
        assert hasattr(model_loaded, "predict_proba"), \
            "Model tidak punya predict_proba!"
        print(f"  [OK] Verifikasi berhasil — model bisa di-load")
    except Exception as e:
        print(f"  [ERROR] Verifikasi gagal: {e}")
        raise
    return str(model_path)

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    random_state: int = RANDOM_STATE,
) -> dict:
    print("\n PERBANDINGAN SEMUA MODEL (sebelum tuning)")
    print("=" * 65)
 
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    model_configs = {
        "Logistic Regression": SKPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=500,
                C=1.0, random_state=random_state,
            )),
        ]),
        "Linear SVM": CalibratedClassifierCV(
            SKPipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(
                    class_weight="balanced", max_iter=2000,
                    C=1.0, random_state=random_state,
                )),
            ]),
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=20, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, num_leaves=31,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20,
            scale_pos_weight=scale, random_state=random_state,
            verbose=-1, n_jobs=-1,
        ),
    }
 
    all_results = {}
    for name, model in model_configs.items():
        print(f"\n  ── {name} {'─'*(45-len(name))}")
        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0
 
        y_pred  = model.predict(X_test)
        y_proba = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, "predict_proba")
                   else model.decision_function(X_test))
 
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc  = average_precision_score(y_test, y_proba)
 
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        cv_s = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
 
        report = classification_report(
            y_test, y_pred,
            target_names=["Non-Duplikat", "Duplikat"],
            digits=4,
        )
        print(report)
        print(f"  ROC-AUC (test) : {roc_auc:.4f}")
        print(f"  PR-AUC  (test) : {pr_auc:.4f}")
        print(f"  3-CV (train)   : {cv_s.mean():.4f} ± {cv_s.std():.4f}")
        print(f"  Waktu fit      : {t_fit:.1f}s")
 
        all_results[name] = {
            "model":   model,
            "name":    name,
            "roc_auc": roc_auc,
            "pr_auc":  pr_auc,
            "cv_mean": float(cv_s.mean()),
            "cv_std":  float(cv_s.std()),
            "fit_time": t_fit,
            "report":  report,
        }
 
    return all_results
 
 
def compare_all_models(
    all_results: dict,
    tuned_metrics: dict,
    save_path: str = "perbandingan_semua_model.png",
) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  PERBANDINGAN SEMUA MODEL")
    print("=" * 65)
 
    rows = []
    for name, r in all_results.items():
        label = f"{name} (baseline)" if name == "LightGBM" else name
        rows.append({
            "Model":          label,
            "ROC-AUC (test)": r["roc_auc"],
            "PR-AUC (test)":  r["pr_auc"],
            "CV mean":        r["cv_mean"],
            "CV std":         r["cv_std"],
            "CV gap":         abs(r["roc_auc"] - r["cv_mean"]),
            "Fit time (s)":   round(r["fit_time"], 1),
        })
 
    rows.append({
        "Model":          "LightGBM (tuned)",
        "ROC-AUC (test)": tuned_metrics["roc_auc"],
        "PR-AUC (test)":  tuned_metrics["pr_auc"],
        "CV mean":        tuned_metrics["cv_mean"],
        "CV std":         tuned_metrics["cv_std"],
        "CV gap":         abs(tuned_metrics["roc_auc"] - tuned_metrics["cv_mean"]),
        "Fit time (s)":   round(tuned_metrics.get("tune_time", 0), 1),
    })
 
    df = pd.DataFrame(rows).sort_values("ROC-AUC (test)", ascending=False)
    df = df.reset_index(drop=True)
 
    df_disp = df.copy()
    for col in ["ROC-AUC (test)", "PR-AUC (test)", "CV mean", "CV std", "CV gap"]:
        df_disp[col] = df_disp[col].map("{:.4f}".format)
    print(df_disp.to_string(index=False))
 
    best_baseline = max(all_results.values(), key=lambda r: r["roc_auc"])
    print(f"\n  Model baseline terbaik : {best_baseline['name']}")
    print(f"    ROC-AUC : {best_baseline['roc_auc']:.4f}")
    print(f"    PR-AUC  : {best_baseline['pr_auc']:.4f}")
    print(f"\n  LightGBM setelah tuning:")
    print(f"    ROC-AUC : {tuned_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC  : {tuned_metrics['pr_auc']:.4f}")
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = sns.color_palette("Set2", len(df))
    colors = [
        "#D85A30" if "tuned" in m else c
        for m, c in zip(df["Model"], palette)
    ]
    names_plot = df["Model"].tolist()
    y_pos      = np.arange(len(names_plot))
 
    bars = axes[0].barh(y_pos, df["ROC-AUC (test)"].values,
                        color=colors, alpha=0.85)
    axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set(yticks=y_pos, yticklabels=names_plot,
                xlabel="ROC-AUC", title="ROC-AUC per Model",
                xlim=[0.4, 1.05])
    for bar, val in zip(bars, df["ROC-AUC (test)"].values):
        axes[0].text(val + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=8)
    axes[0].title.set_fontweight("bold")
 
    bars = axes[1].barh(y_pos, df["PR-AUC (test)"].values,
                        color=colors, alpha=0.85)
    axes[1].set(yticks=y_pos, yticklabels=names_plot,
                xlabel="PR-AUC", title="PR-AUC per Model",
                xlim=[0, 1.05])
    for bar, val in zip(bars, df["PR-AUC (test)"].values):
        axes[1].text(val + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=8)
    axes[1].title.set_fontweight("bold")
 
    axes[2].barh(y_pos, df["CV mean"].values,
                 xerr=df["CV std"].values,
                 color=colors, alpha=0.85, capsize=4)
    axes[2].set(yticks=y_pos, yticklabels=names_plot,
                xlabel="CV ROC-AUC (mean ± std)",
                title="3-Fold CV ROC-AUC",
                xlim=[0.4, 1.05])
    for i, (mean, std) in enumerate(zip(df["CV mean"].values,
                                        df["CV std"].values)):
        axes[2].text(mean + std + 0.01, i,
                     f"{mean:.4f}±{std:.4f}", va="center", fontsize=7)
    axes[2].title.set_fontweight("bold")
 
    plt.suptitle(
        "Perbandingan Semua Model  | LightGBM setelah tuning (disimpan)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  [OK] Plot perbandingan: {save_path}")
 
    return df

# MAIN
def main():
    t_start = time.time()
    print("KLASIFIKASI DUPLIKASI LAPORAN")
    print(f"\n  Konfigurasi:")
    print(f"    Label dup  : jarak < {LABEL_KM*1000:.0f}m DAN waktu < {LABEL_JAM:.0f} jam")
    print(f"    Pairing    : ≤ {PAIRING_KM*1000:.0f}m, ≤ {PAIRING_JAM:.0f} jam")
    print(f"    Test size  : {TEST_SIZE:.0%}")
    print(f"    Fitur      : {FEATURE_NAMES}")

    DATA_PATH = "dataset_2024_final.csv"
    df_raw    = load_and_validate(DATA_PATH)

    df_train_raw, df_test_raw = split_at_report_level(df_raw)

    print("\n  Encoding kategori")
    label_enc = LabelEncoder()
    label_enc.fit(df_train_raw["DESCRIPTION_GROUPED"])
    print(f"  Kategori dikenal: {list(label_enc.classes_)}")

    X_train, y_train = generate_pairs_features_labels(
        df_train_raw, label_enc, set_name="train"
    )

    X_test, y_test = generate_pairs_features_labels(
        df_test_raw, label_enc, set_name="test"
    )

    plot_eda(X_train, y_train, save_path="eda_fitur.png")

    all_results = train_all_models(X_train, y_train, X_test, y_test)

    baseline_model, baseline_metrics = train_baseline_lightgbm(
        X_train, y_train, X_test, y_test
    )

    tuned_model, best_params, tuned_metrics = tune_lightgbm(
        X_train, y_train, X_test, y_test,
        n_iter=30, cv_folds=3,
    )

    df_comparison = compare_all_models(all_results, tuned_metrics)

    if tuned_metrics["roc_auc"] >= baseline_metrics["roc_auc"]:
        final_model   = tuned_model
        final_metrics = tuned_metrics
        print("\n  Model terbaik: TUNED")
    else:
        final_model   = baseline_model
        final_metrics = baseline_metrics
        best_params   = {}
        print("\n  Model terbaik: BASELINE (tuning tidak meningkatkan performa)")

    save_model(final_model, best_params, final_metrics, output_dir=".")

    df_train_out = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    df_train_out["label"] = y_train
    df_train_out.to_csv("data_train_pairs.csv", index=False)

    df_test_out = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    df_test_out["label"] = y_test
    df_test_out.to_csv("data_test_pairs.csv", index=False)

    print(f"  [OK] Data train : data_train_pairs.csv  ({len(df_train_out):,} rows)")
    df = pd.read_csv("data_train_pairs.csv")
    print(df.head(10))
    print(f"  [OK] Data test  : data_test_pairs.csv   ({len(df_test_out):,} rows)")
    df = pd.read_csv("data_test_pairs.csv")
    print(df.head(10))

    t_total = time.time() - t_start
    print("  PIPELINE SELESAI")
    print(f"  Total waktu   : {t_total:.0f} detik")
    return {
        "df_train_raw":     df_train_raw,
        "df_test_raw":      df_test_raw,
        "X_train":          X_train,
        "y_train":          y_train,
        "X_test":           X_test,
        "y_test":           y_test,
        "baseline_model":   baseline_model,
        "tuned_model":      tuned_model,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics":    tuned_metrics,
        "best_params":      best_params,
    }
 

if __name__ == "__main__":
    out = main()