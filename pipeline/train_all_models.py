"""
3-Model Training Pipeline — corrected per session plan
  Tab 1: Host Churn         — real-estate-analytics churn dataset + SMOTE
  Tab 2: Social Media       — Instagram Analytics (viral/high vs low/medium)
  Tab 3: Listing Sold       — listings + properties joined, sold_flag target
"""
import json, warnings, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
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
DS    = BASE / "datasets" / "260515"
RE_DS = DS / "real-estate-analytics-revenue-behavior-and-churn"
MODEL = BASE / "webapp" / "model"
MODEL.mkdir(parents=True, exist_ok=True)

CLASSIFIERS = {
    "LogisticRegression":   LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree":         DecisionTreeClassifier(random_state=42),
    "RandomForest":         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting":     GradientBoostingClassifier(n_estimators=150, random_state=42),
    "XGBoost":              XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, n_estimators=150),
    "KNN":                  KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "GaussianNB":           GaussianNB(),
    "AdaBoost":             AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVC":                  SVC(probability=True, kernel="rbf", random_state=42),
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
            proba = clf.predict_proba(X_te)[:,1]
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
# TAB 1: Host Churn Predictor
# Dataset: customer_monthly_metrics (churn labels) + customers (demographics)
# SMOTE: yes — only 2.85% positive class
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAB 1: Host Churn Predictor")
print("=" * 60)

customers = pd.read_csv(RE_DS / "customers.csv")
monthly   = pd.read_csv(RE_DS / "customer_monthly_metrics.csv")

# Aggregate monthly features per customer
agg = monthly.groupby("customer_id").agg(
    avg_calls        = ("calls",          "mean"),
    avg_views        = ("views",          "mean"),
    avg_visits       = ("visits",         "mean"),
    avg_sessions     = ("sessions",       "mean"),
    avg_session_min  = ("avg_session_min","mean"),
    avg_engagement   = ("engagement_score","mean"),
    avg_churn_risk   = ("churn_risk",     "mean"),
    avg_revenue      = ("revenue",        "mean"),
    total_deals      = ("deals",          "sum"),
    months_active    = ("month",          "count"),
    churned          = ("churned",        "max"),   # True if ever churned
).reset_index()

# Merge with customer demographics
df1 = agg.merge(customers[["customer_id","segment","acquisition_channel",
                            "income_band","age","household_size","propensity_score"]],
                on="customer_id", how="left")

df1["churned"] = df1["churned"].map({True:1, False:0})
print(f"  Rows: {len(df1):,}  Churn rate: {df1['churned'].mean()*100:.1f}%")

le_seg  = LabelEncoder(); df1["segment_enc"]     = le_seg.fit_transform(df1["segment"].fillna("End-User"))
le_acq  = LabelEncoder(); df1["acquisition_enc"] = le_acq.fit_transform(df1["acquisition_channel"].fillna("Organic"))
le_inc  = LabelEncoder(); df1["income_enc"]      = le_inc.fit_transform(df1["income_band"].fillna("Mid"))

F1 = ["propensity_score","segment_enc","acquisition_enc","income_enc",
      "age","household_size","avg_calls","avg_views","avg_visits",
      "avg_sessions","avg_session_min","avg_engagement","avg_churn_risk",
      "avg_revenue","total_deals","months_active"]

X1 = df1[F1].fillna(0)
y1 = df1["churned"]

# SMOTE to balance the severe class imbalance
print(f"  Before SMOTE — 0:{(y1==0).sum()}  1:{(y1==1).sum()}")
sm = SMOTE(random_state=42, k_neighbors=5)
X1_res, y1_res = sm.fit_resample(X1, y1)
print(f"  After  SMOTE — 0:{(y1_res==0).sum()}  1:{(y1_res==1).sum()}")
X1_res = pd.DataFrame(X1_res, columns=F1)

encoders1 = {
    "segments":     list(le_seg.classes_),
    "channels":     list(le_acq.classes_),
    "income_bands": list(le_inc.classes_),
}
r1, s1, n1 = train_and_save("churn", X1_res, y1_res, F1, encoders1)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Social Media Content Predictor
# Dataset: Instagram Analytics (30k) — balanced 4-class, binarise to high-perf
# Target: viral or high = 1, medium or low = 0 (50/50 split)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAB 2: Social Media Content Predictor")
print("=" * 60)

ig = pd.read_csv(DS / "instagram-analytics-dataset" / "Instagram_Analytics.csv")
print(f"  Rows: {ig.shape[0]:,}")
print(f"  Performance distribution:\n{ig['performance_bucket_label'].value_counts()}")

# Binary: viral/high = 1 (top-performing content that drives leads/attention)
ig["target"] = ig["performance_bucket_label"].isin(["viral","high"]).astype(int)
print(f"  High-performance: {ig['target'].sum():,} / {len(ig):,} ({ig['target'].mean()*100:.0f}%)")

le_at  = LabelEncoder(); ig["account_type_enc"]     = le_at.fit_transform(ig["account_type"].fillna("brand"))
le_mt  = LabelEncoder(); ig["media_type_enc"]       = le_mt.fit_transform(ig["media_type"].fillna("post"))
le_cc  = LabelEncoder(); ig["content_category_enc"] = le_cc.fit_transform(ig["content_category"].fillna("Lifestyle"))
le_ts  = LabelEncoder(); ig["traffic_source_enc"]   = le_ts.fit_transform(ig["traffic_source"].fillna("Home Feed"))
le_dow = LabelEncoder(); ig["day_enc"]              = le_dow.fit_transform(ig["day_of_week"].fillna("Monday"))

F2 = ["account_type_enc","media_type_enc","content_category_enc","traffic_source_enc",
      "follower_count","has_call_to_action","post_hour","day_enc",
      "caption_length","hashtags_count"]

X2 = ig[F2].fillna(0)
y2 = ig["target"]

encoders2 = {
    "account_types":    list(le_at.classes_),
    "media_types":      list(le_mt.classes_),
    "content_cats":     list(le_cc.classes_),
    "traffic_sources":  list(le_ts.classes_),
    "days":             list(le_dow.classes_),
}
r2, s2, n2 = train_and_save("social", X2, y2, F2, encoders2)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Listing Performance Predictor
# Dataset: listings.csv + properties.csv joined on property_id
# Target: sold_flag (True/False)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TAB 3: Listing Performance Predictor")
print("=" * 60)

listings   = pd.read_csv(RE_DS / "listings.csv")
properties = pd.read_csv(RE_DS / "properties.csv")

df3 = listings.merge(properties, on="property_id", how="left")
df3["sold"] = df3["sold_flag"].map({True:1, False:0})
print(f"  Rows: {len(df3):,}  Sold: {df3['sold'].mean()*100:.1f}%")

le_pt  = LabelEncoder(); df3["prop_type_enc"]    = le_pt.fit_transform(df3["property_type"].fillna("Apartment"))
le_lc  = LabelEncoder(); df3["list_channel_enc"] = le_lc.fit_transform(df3["listing_channel"].fillna("Open"))
le_c3  = LabelEncoder(); df3["city_enc"]         = le_c3.fit_transform(df3["city"].fillna("Unknown"))

F3 = ["prop_type_enc","size_sqm","bedrooms","bathrooms","year_built",
      "location_score","amenities_count","list_price","has_parking",
      "near_transit","near_school","list_channel_enc","city_enc","listed_price"]

for col in ["size_sqm","bedrooms","bathrooms","year_built","location_score",
            "amenities_count","list_price","listed_price"]:
    df3[col] = pd.to_numeric(df3[col], errors="coerce")

for col in ["has_parking","near_transit","near_school"]:
    df3[col] = df3[col].map({True:1,False:0,"True":1,"False":0}).fillna(0)

X3 = df3[F3].fillna(df3[F3].median())
y3 = df3["sold"]

encoders3 = {
    "property_types": list(le_pt.classes_),
    "list_channels":  list(le_lc.classes_),
    "cities":         list(le_c3.classes_),
}
r3, s3, n3 = train_and_save("listing", X3, y3, F3, encoders3)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL MODELS TRAINED")
print(f"  Tab 1 Churn    — Best: {n1}  F1={s1:.4f}")
print(f"  Tab 2 Social   — Best: {n2}  F1={s2:.4f}")
print(f"  Tab 3 Listing  — Best: {n3}  F1={s3:.4f}")
print("=" * 60)
