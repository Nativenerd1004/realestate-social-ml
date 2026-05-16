"""
Retrain Tab 2: Property Market Position Predictor — USA + Canada
Target: Is property price above the state/province median?
US: USA Real Estate Dataset (sample 200k from 1.47M)
CA: Canadian House Prices for Top Cities (35k rows, 9 provinces)
Combined: 235k rows across 63 states/provinces
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
    X_s = scaler.fit_transform(X)
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
            results[mname] = {"accuracy": round(acc, 4), "precision": round(prec, 4),
                              "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": round(roc, 4)}
            print(f"  {mname:22s}  F1={f1:.4f}  ROC={roc:.4f}  Acc={acc:.4f}", flush=True)
            if f1 > best_f1:
                best_f1, best, best_name = f1, clf, mname
        except Exception as e:
            print(f"  {mname} FAILED: {e}", flush=True)

    print(f"\n  ✓ Best [{name}]: {best_name}  F1={best_f1:.4f}\n", flush=True)
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


print("=" * 60, flush=True)
print("TAB 2: Property Market Position Predictor (USA + Canada)", flush=True)
print("=" * 60, flush=True)

# ── Load US data ──────────────────────────────────────────────────────────
print("  Loading USA Real Estate Dataset...", flush=True)
us = pd.read_csv(DS / "usa-real-estate-dataset" / "USA Real Estate Dataset.csv", low_memory=False)
us = us[(us["price"] > 10_000) & (us["price"] < 5_000_000)].copy()
us = us[(us["bed"].between(1, 15)) & (us["bath"].between(1, 15))].copy()
us = us[(us["house_size"] > 100) & (us["house_size"] < 20_000)].copy()
us["acre_lot"] = us["acre_lot"].clip(0, 50)
us = us.dropna(subset=["price", "bed", "bath", "house_size", "state", "area_type"])
us["country_flag"] = "US"
us_sample = us.sample(n=200_000, random_state=42)
print(f"  US: {len(us_sample):,} rows (sampled from {len(us):,})", flush=True)

# US medians for imputing Canada missing fields
us_house_size_median = us_sample["house_size"].median()
us_acre_lot_median   = us_sample["acre_lot"].median()
print(f"  US medians — house_size={us_house_size_median:.0f} sqft, acre_lot={us_acre_lot_median:.4f}", flush=True)

# ── Load Canada data ──────────────────────────────────────────────────────
print("  Loading Canada House Prices...", flush=True)
ca = pd.read_csv(DS / "canadian-house-prices" / "HouseListings-Top45Cities-10292023-kaggle.csv",
                 encoding="latin-1")

# Standardize column names to match US
ca = ca.rename(columns={"Number_Beds": "bed", "Number_Baths": "bath",
                         "Province": "state", "Price": "price"})

# Filter outliers
ca = ca[(ca["price"] > 10_000) & (ca["price"] < 10_000_000)].copy()
ca = ca[(ca["bed"].between(0, 15)) & (ca["bath"].between(0, 15))].copy()
ca = ca.dropna(subset=["price", "bed", "bath", "state"])

# Engineer area_type from city Population
def pop_to_area(pop):
    if pop >= 500_000:
        return "urban"
    elif pop >= 100_000:
        return "suburban"
    else:
        return "rural"

ca["area_type"] = ca["Population"].apply(pop_to_area)

# Impute missing house_size and acre_lot with US medians
ca["house_size"] = us_house_size_median
ca["acre_lot"]   = us_acre_lot_median
ca["country_flag"] = "CA"
print(f"  Canada: {len(ca):,} rows, Provinces: {sorted(ca['state'].unique())}", flush=True)

# ── Combine US + Canada ───────────────────────────────────────────────────
combined = pd.concat([
    us_sample[["price", "bed", "bath", "acre_lot", "house_size", "state", "area_type", "country_flag"]],
    ca[["price", "bed", "bath", "acre_lot", "house_size", "state", "area_type", "country_flag"]],
], ignore_index=True)
print(f"  Combined: {len(combined):,} rows", flush=True)

# Target: above state/province median price
state_medians   = combined.groupby("state")["price"].median()
combined["state_median"]  = combined["state"].map(state_medians)
combined["above_market"]  = (combined["price"] > combined["state_median"]).astype(int)
print(f"  Above-market: {combined['above_market'].mean()*100:.1f}%", flush=True)

# ── Encode ────────────────────────────────────────────────────────────────
le_area  = LabelEncoder()
le_state = LabelEncoder()
combined["area_enc"]  = le_area.fit_transform(combined["area_type"].fillna("suburban"))
combined["state_enc"] = le_state.fit_transform(combined["state"].fillna("Unknown"))

all_locations = list(le_state.classes_)
print(f"  Locations: {len(all_locations)} (US states + CA provinces)", flush=True)

# Save encoder files (same keys as before — webapp reads these)
(MODEL / "social_area_types.json").write_text(json.dumps(list(le_area.classes_)))
(MODEL / "social_states.json").write_text(json.dumps(all_locations))

F2 = ["bed", "bath", "acre_lot", "house_size", "area_enc", "state_enc"]
X2 = combined[F2].fillna(combined[F2].median())
y2 = combined["above_market"]
print(f"  Training on {len(X2):,} samples", flush=True)

r2, s2, n2 = train_and_save("social", X2, y2, F2, {
    "area_types": list(le_area.classes_),
    "states":     all_locations,
})

print("=" * 60, flush=True)
print(f"DONE — Tab 2 Market Position (USA+Canada): {n2}  F1={s2:.4f}", flush=True)
print("=" * 60, flush=True)
