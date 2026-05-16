"""
Retrain Tab 2 (Social Media) and Tab 3 (Listing Sale) with better datasets.
Tab 2: Social Media Advertising (300k) — Conversion_Rate > median → high lead potential
Tab 3: Illinois Real Estate 2026 (8.5k)  — sold_to_list_ratio >= 1.0 → sold at/above asking
"""
import json, warnings, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
    "DecisionTree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
    "XGBoost":            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, n_estimators=150, max_depth=6),
    "KNN":                KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "GaussianNB":         GaussianNB(),
    "AdaBoost":           AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVC":                SVC(probability=True, kernel="rbf", random_state=42),
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Social Media Advertising → Lead Conversion Predictor
# Dataset: Social_Media_Advertising.csv (300,000 rows)
# Target: Conversion_Rate > median (0.08) = high lead potential
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAB 2: Social Media Lead Conversion Predictor")
print("=" * 60)

df2 = pd.read_csv(DS / "social-media-advertising-dataset" / "Social_Media_Advertising.csv",
                  low_memory=False)
print(f"  Rows: {len(df2):,}")

median_cr = df2["Conversion_Rate"].median()
df2["target"] = (df2["Conversion_Rate"] > median_cr).astype(int)
print(f"  Median Conversion_Rate: {median_cr}  →  High: {df2['target'].sum():,} / {len(df2):,} ({df2['target'].mean()*100:.0f}%)")

# Encode categoricals
le_goal = LabelEncoder(); df2["goal_enc"]     = le_goal.fit_transform(df2["Campaign_Goal"].fillna("Brand Awareness"))
le_ch   = LabelEncoder(); df2["channel_enc"]  = le_ch.fit_transform(df2["Channel_Used"].fillna("Instagram"))
le_loc  = LabelEncoder(); df2["loc_enc"]      = le_loc.fit_transform(df2["Location"].fillna("USA"))
le_lang = LabelEncoder(); df2["lang_enc"]     = le_lang.fit_transform(df2["Language"].fillna("English"))
le_seg  = LabelEncoder(); df2["seg_enc"]      = le_seg.fit_transform(df2["Customer_Segment"].fillna("General"))
le_aud  = LabelEncoder(); df2["aud_enc"]      = le_aud.fit_transform(df2["Target_Audience"].fillna("All"))

(MODEL / "social_goals.json").write_text(json.dumps(list(le_goal.classes_)))
(MODEL / "social_channels.json").write_text(json.dumps(list(le_ch.classes_)))
(MODEL / "social_locations.json").write_text(json.dumps(list(le_loc.classes_)))
(MODEL / "social_languages.json").write_text(json.dumps(list(le_lang.classes_)))
(MODEL / "social_segments.json").write_text(json.dumps(list(le_seg.classes_)))
(MODEL / "social_audiences.json").write_text(json.dumps(list(le_aud.classes_)))

F2 = ["goal_enc", "channel_enc", "loc_enc", "lang_enc", "seg_enc", "aud_enc",
      "Duration", "Acquisition_Cost", "Clicks", "Impressions", "Engagement_Score"]

for col in ["Duration", "Acquisition_Cost", "Clicks", "Impressions", "Engagement_Score"]:
    df2[col] = pd.to_numeric(df2[col], errors="coerce")

X2 = df2[F2].fillna(df2[F2].median())
y2 = df2["target"]

r2, s2, n2 = train_and_save("social", X2, y2, F2, {
    "goals":     list(le_goal.classes_),
    "channels":  list(le_ch.classes_),
    "locations": list(le_loc.classes_),
    "languages": list(le_lang.classes_),
    "segments":  list(le_seg.classes_),
    "audiences": list(le_aud.classes_),
})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Listing Sale Predictor — Illinois Real Estate 2026
# Target: sold_to_list_ratio >= 1.0 = sold at or above asking price
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAB 3: Listing Sale Predictor (Illinois 2026)")
print("=" * 60)

df3 = pd.read_csv(DS / "illinois-real-estate-sold-properties-data-2026" / "Illinois_real_estate_ultimate.csv",
                  low_memory=False)
print(f"  Rows: {len(df3):,}  Cols: {list(df3.columns)}")

# Cap ratio outliers (max=550k is data error), keep reasonable range
df3 = df3[df3["sold_to_list_ratio"].between(0.5, 2.0)].copy()
df3["target"] = (df3["sold_to_list_ratio"] >= 1.0).astype(int)
print(f"  After outlier cap: {len(df3):,} rows  Sold>=asking: {df3['target'].sum():,} ({df3['target'].mean()*100:.1f}%)")

le_type = LabelEncoder(); df3["type_enc"]     = le_type.fit_transform(df3["type"].fillna("Single Family"))
le_sub  = LabelEncoder(); df3["sub_type_enc"] = le_sub.fit_transform(df3["sub_type"].fillna(""))

(MODEL / "listing_types.json").write_text(json.dumps(list(le_type.classes_)))
(MODEL / "listing_subtypes.json").write_text(json.dumps(list(le_sub.classes_)))

for col in ["listPrice","sqft","lot_sqft","stories","beds","baths","baths_full",
            "garage","year_built","price_per_sqft_sold","property_age_at_sale"]:
    df3[col] = pd.to_numeric(df3[col], errors="coerce")

F3 = ["type_enc","sub_type_enc","listPrice","sqft","lot_sqft","stories",
      "beds","baths","baths_full","garage","year_built",
      "price_per_sqft_sold","property_age_at_sale"]

X3 = df3[F3].fillna(df3[F3].median())
y3 = df3["target"]

r3, s3, n3 = train_and_save("listing", X3, y3, F3, {
    "property_types": list(le_type.classes_),
    "sub_types":      list(le_sub.classes_),
})

print("=" * 60)
print("RETRAINING COMPLETE")
print(f"  Tab 2 Social   — Best: {n2}  F1={s2:.4f}")
print(f"  Tab 3 Listing  — Best: {n3}  F1={s3:.4f}")
print("=" * 60)
