"""
PIPELINE DUPLIKASI LAPORAN — ANTI LEAKAGE (FINAL)

ALUR:
1. Load dataset
2. Split laporan (train/test)
3. Generate pairs TERPISAH
4. Feature engineering
5. Label generation
6. Training & evaluasi
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

PAIRING_JARAK_KM = 0.5
PAIRING_JAM = 6

LABEL_JARAK_KM = 0.05
LABEL_JAM = 2

# ============================================================
# HAVERSINE
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# LOAD
# ============================================================
def load_data(path):
    df = pd.read_csv(path)
    df["ADDDATE"] = pd.to_datetime(df["ADDDATE"], errors="coerce")

    df = df.dropna(subset=["LATITUDE", "LONGITUDE", "ADDDATE"])
    return df.reset_index(drop=True)

# ============================================================
# SPLIT AWAL (ANTI LEAKAGE)
# ============================================================
def split_data(df):
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["DESCRIPTION_GROUPED"]
    )

# ============================================================
# GENERATE PAIRS
# ============================================================
def generate_pairs(df):
    all_pairs = []

    for kategori in df["DESCRIPTION_GROUPED"].unique():
        df_kat = df[df["DESCRIPTION_GROUPED"] == kategori]\
                 .sort_values("ADDDATE").reset_index(drop=True)

        if len(df_kat) < 2:
            continue

        coords = np.radians(df_kat[["LATITUDE", "LONGITUDE"]].values)
        radius = PAIRING_JARAK_KM / 6371.0

        nn = NearestNeighbors(radius=radius, metric="haversine")
        nn.fit(coords)

        neighbors = nn.radius_neighbors(coords, return_distance=False)

        for i, neigh in enumerate(neighbors):
            for j in neigh:
                if j <= i:
                    continue

                t1 = df_kat.iloc[i]["ADDDATE"]
                t2 = df_kat.iloc[j]["ADDDATE"]

                diff_jam = abs((t2 - t1).total_seconds()) / 3600
                if diff_jam > PAIRING_JAM:
                    continue

                all_pairs.append({
                    "lat1": df_kat.iloc[i]["LATITUDE"],
                    "lon1": df_kat.iloc[i]["LONGITUDE"],
                    "lat2": df_kat.iloc[j]["LATITUDE"],
                    "lon2": df_kat.iloc[j]["LONGITUDE"],
                    "t1": t1,
                    "t2": t2,
                    "kategori": kategori
                })

    return pd.DataFrame(all_pairs)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineering(df):
    df["selisih_jarak"] = haversine_km(
        df["lat1"], df["lon1"], df["lat2"], df["lon2"]
    )

    df["selisih_jam"] = (
        (df["t2"] - df["t1"]).abs().dt.total_seconds() / 3600
    )

    df["log_jarak"] = np.log1p(df["selisih_jarak"])
    df["log_jam"] = np.log1p(df["selisih_jam"])

    return df

# ============================================================
# LABEL
# ============================================================
def generate_label(df):
    df["label"] = (
        (df["selisih_jarak"] < LABEL_JARAK_KM) &
        (df["selisih_jam"] < LABEL_JAM)
    ).astype(int)
    return df

# ============================================================
# PREPARE
# ============================================================
def prepare_xy(df):
    X = df[["log_jarak", "log_jam"]]
    y = df["label"]
    return X, y

# ============================================================
# TRAIN
# ============================================================
def train_model(X_train, y_train, X_test, y_test, name):
    if name == "lr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced"))
        ])
    elif name == "svm":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC())
        ])
        model = CalibratedClassifierCV(base, cv=2)
    elif name == "rf":
        model = RandomForestClassifier(n_estimators=100)
    elif name == "xgb":
        model = XGBClassifier(eval_metric="logloss")
    else:
        model = LGBMClassifier()

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:,1] \
        if hasattr(model, "predict_proba") \
        else model.decision_function(X_test)

    pred = model.predict(X_test)

    print(f"\n=== {name.upper()} ===")
    print(classification_report(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

# ============================================================
# MAIN
# ============================================================
def main():
    df = load_data("dataset_2024_final.csv")

    print("Split data...")
    df_train, df_test = split_data(df)

    print("Generate pairs...")
    train_pairs = generate_pairs(df_train)
    test_pairs  = generate_pairs(df_test)

    print("Feature engineering...")
    train_pairs = feature_engineering(train_pairs)
    test_pairs  = feature_engineering(test_pairs)

    print("Labeling...")
    train_pairs = generate_label(train_pairs)
    test_pairs  = generate_label(test_pairs)

    X_train, y_train = prepare_xy(train_pairs)
    X_test, y_test   = prepare_xy(test_pairs)

    for m in ["lr", "svm", "rf", "xgb", "lgb"]:
        train_model(X_train, y_train, X_test, y_test, m)

if __name__ == "__main__":
    main()