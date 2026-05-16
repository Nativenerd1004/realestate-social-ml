"""
Tab 4: Social Media Content Performance Predictor
Dataset: YouTube Trending Videos Stats 2026 (178k rows, 11 countries)
Target: like_rate (likes/views) > median → high engagement vs low
Features: category, country, title_length, tag_count, publish_hour, publish_dow, log_views, comment_rate
"""
import json, sys, warnings, joblib, glob, os
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
DS    = BASE / "datasets" / "260516" / "youtube-trending-2026"
MODEL = BASE / "webapp" / "model"
MODEL.mkdir(parents=True, exist_ok=True)

# YouTube category_id → human-readable label
YT_CATEGORIES = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
    15: "Pets & Animals", 17: "Sports", 18: "Short Movies",
    19: "Travel & Events", 20: "Gaming", 21: "Videoblogging",
    22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News & Politics", 26: "Howto & Style", 27: "Education",
    28: "Science & Technology", 29: "Nonprofits & Activism",
}

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
print("TAB 4: Social Media Content Performance Predictor", flush=True)
print("(YouTube Trending 2026 — 11 countries, 178k rows)", flush=True)
print("=" * 60, flush=True)

# Load all country files
files = sorted(glob.glob(str(DS / "*_Trending.csv")))
dfs = []
for f in files:
    country = os.path.basename(f).split("_")[0]
    df = pd.read_csv(f, low_memory=False)
    df["country"] = country
    dfs.append(df)
yt = pd.concat(dfs, ignore_index=True)
print(f"  Loaded: {len(yt):,} rows from {len(files)} country files", flush=True)

# Drop rows with 0 views (can't compute like-rate)
yt = yt[yt["views"] > 0].copy()

# Feature engineering
yt["like_rate"]    = yt["likes"] / yt["views"]
yt["comment_rate"] = yt["comments"] / yt["views"]
yt["log_views"]    = np.log1p(yt["views"])
yt["title_length"] = yt["title"].fillna("").apply(len)
yt["tag_count"]    = yt["tags"].fillna("").apply(lambda t: len(t.split("|")) if t and t != "[none]" else 0)

# Parse publish_time
yt["publish_time"] = pd.to_datetime(yt["publish_time"], errors="coerce", utc=True)
yt["publish_hour"] = yt["publish_time"].dt.hour.fillna(12).astype(int)
yt["publish_dow"]  = yt["publish_time"].dt.dayofweek.fillna(0).astype(int)

# Target: like_rate > median
median_lr = yt["like_rate"].median()
yt["target"] = (yt["like_rate"] > median_lr).astype(int)
print(f"  Like-rate median={median_lr:.5f}  High engagement: {yt['target'].mean()*100:.1f}%", flush=True)

# Encode categoricals
le_cat     = LabelEncoder()
le_country = LabelEncoder()
yt["cat_enc"]     = le_cat.fit_transform(yt["category_id"].fillna(24).astype(int).astype(str))
yt["country_enc"] = le_country.fit_transform(yt["country"])

# Save encoder lists (human-readable category names)
cat_labels = [YT_CATEGORIES.get(int(c), f"Category {c}") for c in le_cat.classes_]
(MODEL / "yt_categories.json").write_text(json.dumps(cat_labels))
(MODEL / "yt_category_ids.json").write_text(json.dumps(list(le_cat.classes_)))
(MODEL / "yt_countries.json").write_text(json.dumps(list(le_country.classes_)))

F4 = ["cat_enc", "country_enc", "title_length", "tag_count",
      "publish_hour", "publish_dow", "log_views", "comment_rate"]

X4 = yt[F4].fillna(yt[F4].median())
y4 = yt["target"]
print(f"  Training on {len(X4):,} samples", flush=True)

r4, s4, n4 = train_and_save("yt", X4, y4, F4, {
    "categories": cat_labels,
    "category_ids": list(le_cat.classes_),
    "countries":  list(le_country.classes_),
})

print("=" * 60, flush=True)
print(f"DONE — Tab 4 Social Media: {n4}  F1={s4:.4f}", flush=True)
print("=" * 60, flush=True)
