"""
Retrain Tab 2 only — USA Real Estate Market Position Predictor
Target: Is property price above the state median? (above-market vs below-market)
Dataset: USA Real Estate Dataset (sample 200k from 1.47M real rows)
"""
import json, sys, warnings, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE  = Path("/Users/apple/Desktop/RealEstate_SocialMedia_ML")
DS    = BASE / "datasets" / "260516"
MODEL = BASE / "webapp" / "model"
MODEL.mkdir(parents=True, exist_ok=True)

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost":            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, n_estimators=150, max_depth=8),
    "KNN":                KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    "GaussianNB":         GaussianNB(),
    "AdaBoost":           AdaBoostClassifier(n_estimators=100, random_state=42),
}


def train_and_save(name, X, y, features, encoders=None):
    scaler = MinMaxScaler()
    X_s    = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)

    best, best_f1, best_name = None, -1, ""
    results = {}
    for mname, clf in CLASSIFIERS.items():
        try:
            clf.fit(X_tr, y_tr)
            pred  = clf.predict(X_te)
            proba = clf.predict_proba(X_te)[:, 1]
            f1    = f1_score(y_te, pred, zero_division=0)
            roc   = roc_auc_score(y_te, proba)
            acc   = accuracy_score(y_te, pred)
            prec  = precision_score(y_te, pred, zero_division=0)
            rec   = recall_score(y_te, pred, zero_division=0)
            results[mname] = {"accuracy": round(acc,4), "precision": round(prec,4),
                              "recall": round(rec,4), "f1": round(f1,4), "roc_auc": round(roc,4)}
            print(f"  {mname:22s}  F1={f1:.4f}  ROC={roc:.4f}  Acc={acc:.4f}")
            if f1 > best_f1:
                best_f1, best, best_name = f1, clf, mname
        except Exception as e:
            print(f"  {mname} FAILED: {e}")

    print(f"\n  ✓ Best [{name}]: {best_name}  F1={best_f1:.4f}\n")
    joblib.dump(best,   MODEL / f"{name}_model.pkl",  compress=3)
    joblib.dump(scaler, MODEL / f"{name}_scaler.pkl", compress=3)
    pd.DataFrame([X.iloc[0].to_dict()]).to_csv(MODEL / f"{name}_X_ref.csv", index=False)
    (MODEL / f"{name}_features.json").write_text(json.dumps(features))
    (MODEL / f"{name}_scores.json").write_text(json.dumps(results, indent=2))
    (MODEL / f"{name}_best.json").write_text(json.dumps({"model": best_name, "f1": best_f1}))
    if encoders:
        for k, v in encoders.items():
            (MODEL / f"{name}_{k}.json").write_text(json.dumps(v))
    return results, best_f1, best_name


print("=" * 60)
print("TAB 2: Property Market Position Predictor")
print("(USA Real Estate — above/below state median price)")
print("=" * 60)

df = pd.read_csv(DS / "usa-real-estate-dataset" / "USA Real Estate Dataset.csv", low_memory=False)
print(f"  Full dataset: {len(df):,} rows")

# Remove obvious outliers
df = df[(df["price"] > 10_000) & (df["price"] < 5_000_000)].copy()
df = df[(df["bed"].between(1, 15)) & (df["bath"].between(1, 15))].copy()
df = df[(df["house_size"] > 100) & (df["house_size"] < 20_000)].copy()
df["acre_lot"] = df["acre_lot"].clip(0, 50)
df = df.dropna(subset=["price", "bed", "bath", "house_size", "state", "area_type"])

# Target: above state median price
state_medians = df.groupby("state")["price"].median()
df["state_median"] = df["state"].map(state_medians)
df["above_market"] = (df["price"] > df["state_median"]).astype(int)
print(f"  After cleaning: {len(df):,}  Above-market: {df['above_market'].mean()*100:.1f}%")

# Sample 200k for training speed
df_sample = df.sample(n=min(200_000, len(df)), random_state=42)

# Encode
le_area  = LabelEncoder(); df_sample["area_enc"]  = le_area.fit_transform(df_sample["area_type"].fillna("suburban"))
le_state = LabelEncoder(); df_sample["state_enc"] = le_state.fit_transform(df_sample["state"].fillna("Unknown"))

(MODEL / "social_area_types.json").write_text(json.dumps(list(le_area.classes_)))
(MODEL / "social_states.json").write_text(json.dumps(list(le_state.classes_)))

F2 = ["bed", "bath", "acre_lot", "house_size", "area_enc", "state_enc"]

X2 = df_sample[F2].fillna(df_sample[F2].median())
y2 = df_sample["above_market"]
print(f"  Training on {len(X2):,} samples  Above market: {y2.mean()*100:.1f}%")

r2, s2, n2 = train_and_save("social", X2, y2, F2, {
    "area_types": list(le_area.classes_),
    "states":     list(le_state.classes_),
})

print("=" * 60)
print(f"DONE — Tab 2 Market Position: {n2}  F1={s2:.4f}")
print("=" * 60)
