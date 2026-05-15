"""
Step 08 — Jupyter Notebook Generator
Auto-generates a clean, presentation-ready .ipynb from the pipeline results.
"""
import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED, MODELS_DIR, NOTEBOOKS, PLOTS, REPORTS, TARGET_COLUMN


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def latest_iteration() -> int:
    files = list(MODELS_DIR.glob("scores_iter*.json"))
    return max((int(f.stem.split("iter")[1]) for f in files), default=1)


def load_best_report() -> dict:
    files = sorted(REPORTS.glob("report_iter*.json"))
    if not files:
        return {}
    with open(files[-1]) as f:
        return json.load(f)


def run():
    print("=" * 60)
    print("STEP 08 — Jupyter Notebook Generator")
    print("=" * 60)

    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    iteration = latest_iteration()
    report    = load_best_report()
    best_model = report.get("best_model", "XGBClassifier")
    best_f1    = report.get("best_f1", "N/A")
    best_roc   = report.get("best_roc_auc", "N/A")

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    cells = [
        # ── Title ──────────────────────────────────────────────────────────────
        md(f"""# Real Estate + Social Media — Churn Prediction
### End-to-End Machine Learning Pipeline
**Best Model:** `{best_model}` | **F1 Score:** `{best_f1}` | **ROC-AUC:** `{best_roc}`
**Iteration:** {iteration} | **Framework:** Samuel's ML Pipeline (STAR Method)

---
"""),

        # ── 1. Imports ─────────────────────────────────────────────────────────
        md("## 1. Imports & Setup"),
        code("""import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib, json
from pathlib import Path

BASE = Path("/Users/apple/Desktop/RealEstate_SocialMedia_ML")
print("Setup complete ✓")
"""),

        # ── 2. Data Overview ───────────────────────────────────────────────────
        md("## 2. Dataset Overview"),
        code(f"""df_raw = pd.read_csv(BASE / "cleaned/ml_ready.csv")
print(f"Shape: {{df_raw.shape}}")
df_raw.head()
"""),
        code("""df_raw.describe()
"""),
        code("""# Missing values check
missing = df_raw.isnull().sum()
missing[missing > 0] if missing.any() else print("No missing values ✓")
"""),

        # ── 3. Target Distribution ─────────────────────────────────────────────
        md(f"## 3. Target Distribution — `{TARGET_COLUMN}`"),
        code(f"""fig, ax = plt.subplots(figsize=(6, 4))
df_raw['{TARGET_COLUMN}'].value_counts().plot(kind='bar', color=['#238636','#da3633'], ax=ax)
ax.set_title('Churn Distribution')
ax.set_xlabel('Churn')
ax.set_ylabel('Count')
plt.xticks([0,1], ['No Churn','Churn'], rotation=0)
plt.tight_layout()
plt.show()
"""),

        # ── 4. Correlation ─────────────────────────────────────────────────────
        md("## 4. Correlation Heatmap"),
        code("""num_df = df_raw.select_dtypes(include=np.number)
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
"""),

        # ── 5. Feature Engineering ─────────────────────────────────────────────
        md("## 5. Feature Engineering & Selection (MinMaxScaler)"),
        code("""df_scaled = pd.read_csv(BASE / "cleaned/ml_scaled.csv")
with open(BASE / "cleaned/selected_features.json") as f:
    features = json.load(f)
print(f"Selected features: {len(features)}")
print(features)
"""),
        code(f"""X = df_scaled[features]
y = df_scaled['{TARGET_COLUMN}']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {{X_train.shape}} | Test: {{X_test.shape}}")
"""),

        # ── 6. Model Training ──────────────────────────────────────────────────
        md(f"## 6. Model Training — 9 Models (Iteration {iteration})"),
        code(f"""scores_path = BASE / "models/scores_iter{iteration}.json"
with open(scores_path) as f:
    scores = json.load(f)

df_scores = pd.DataFrame([
    {{"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"],
     "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"], "CV-F1": v["cv_f1"]}}
    for k, v in scores.items()
]).sort_values("F1", ascending=False)

df_scores.reset_index(drop=True)
"""),
        code("""# Visualise model comparison
fig, ax = plt.subplots(figsize=(12, 5))
df_scores.set_index("Model")[["F1", "ROC-AUC", "Accuracy"]].plot(
    kind="bar", ax=ax, color=["#58a6ff","#f78166","#3fb950"]
)
ax.set_title("Model Comparison")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""),

        # ── 7. Confusion Matrices ──────────────────────────────────────────────
        md("## 7. Confusion Matrices"),
        code(f"""from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

model_files = sorted((BASE / "models").glob("*_iter{iteration}.pkl"))
for idx, model_path in enumerate(model_files[:9]):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"])
    disp.plot(ax=axes[idx], colorbar=False, cmap="Blues")
    axes[idx].set_title(model_path.stem.replace(f"_iter{iteration}",""))

plt.suptitle("Confusion Matrices — All Models", fontsize=14)
plt.tight_layout()
plt.show()
"""),

        # ── 8. ROC Curves ─────────────────────────────────────────────────────
        md("## 8. ROC Curves"),
        code(f"""fig, ax = plt.subplots(figsize=(10, 7))
ax.plot([0,1],[0,1],'--', color='gray', label='Random')

colors = plt.cm.tab10.colors
model_files = sorted((BASE / "models").glob("*_iter{iteration}.pkl"))
for idx, model_path in enumerate(model_files[:9]):
    model = joblib.load(model_path)
    name  = model_path.stem.replace(f"_iter{iteration}","")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[idx % 10], label=f"{{name}} (AUC={{auc:.3f}})")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()
"""),

        # ── 9. Best Model ──────────────────────────────────────────────────────
        md(f"## 9. Best Model — `{best_model}`"),
        code(f"""best_model_path = sorted((BASE / "models").glob("{best_model}_iter*.pkl"))[-1]
best = joblib.load(best_model_path)
y_pred_best = best.predict(X_test)
y_prob_best = best.predict_proba(X_test)[:,1] if hasattr(best,"predict_proba") else y_pred_best

print(f"Accuracy : {{accuracy_score(y_test, y_pred_best):.4f}}")
print(f"Precision: {{precision_score(y_test, y_pred_best):.4f}}")
print(f"Recall   : {{recall_score(y_test, y_pred_best):.4f}}")
print(f"F1 Score : {{f1_score(y_test, y_pred_best):.4f}}")
print(f"ROC-AUC  : {{roc_auc_score(y_test, y_prob_best):.4f}}")
"""),

        # ── 10. Business Insights ──────────────────────────────────────────────
        md("""## 10. Business Insights

**Key Takeaways:**
- The model identifies high-risk customers before they churn
- Top features reveal which real estate / social media signals matter most
- Targeted retention campaigns can be built around the top predictors
- ROC-AUC close to 1.0 means the model discriminates well between churners and non-churners

**Recommended Actions:**
1. Flag customers above 0.7 predicted churn probability for immediate outreach
2. Focus social media content on the top engagement drivers identified
3. Retrain the model monthly as new listing/lead data arrives
4. A/B test retention offers guided by the feature importances
"""),
    ]

    nb.cells = cells
    out_path = NOTEBOOKS / f"RealEstate_SocialMedia_ML_Pipeline_iter{iteration}.ipynb"
    with open(out_path, "w") as f:
        nbf.write(nb, f)

    print(f"[✓] Notebook saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()
