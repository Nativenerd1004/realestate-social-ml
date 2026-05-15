"""
Step 06 — Evaluate Models + Generate All Visualizations
Produces confusion matrices, ROC curves, pair plots, and a summary report.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                              f1_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED, MODELS_DIR, PLOTS, REPORTS, TARGET_COLUMN

DARK   = "#0d1117"
CARD   = "#161b22"
GRID   = "#30363d"
TEXT   = "#c9d1d9"
ACCENT = "#58a6ff"
GREEN  = "#238636"
RED    = "#da3633"


def latest_iteration() -> int:
    files = list(MODELS_DIR.glob("scores_iter*.json"))
    if not files:
        return 1
    return max(int(f.stem.split("iter")[1]) for f in files)


def load_scores(iteration: int) -> dict:
    path = MODELS_DIR / f"scores_iter{iteration}.json"
    with open(path) as f:
        return json.load(f)


def load_test_data():
    X_test = pd.read_csv(CLEANED / "X_test.csv")
    y_test = pd.read_csv(CLEANED / "y_test.csv").squeeze()
    return X_test, y_test


def plot_model_comparison(scores: dict, iteration: int):
    rows = []
    for name, s in scores.items():
        rows.append({"Model": name, "F1": s["f1"], "ROC-AUC": s["roc_auc"],
                     "Accuracy": s["accuracy"], "CV F1": s["cv_f1"]})
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)

    fig = go.Figure()
    for metric, color in [("F1", ACCENT), ("ROC-AUC", "#f78166"), ("Accuracy", "#3fb950"), ("CV F1", "#d2a8ff")]:
        fig.add_trace(go.Bar(name=metric, x=df["Model"], y=df[metric], marker_color=color))

    fig.update_layout(
        barmode="group",
        title=f"Model Comparison — Iteration {iteration}",
        paper_bgcolor=DARK, plot_bgcolor=DARK, font_color=TEXT,
        xaxis_tickangle=-30, legend_bgcolor=CARD,
    )
    out = PLOTS / f"04_model_comparison_iter{iteration}.html"
    fig.write_html(str(out))
    print(f"    [✓] {out.name}")
    return df


def plot_confusion_matrices(scores: dict, X_test, y_test, iteration: int):
    model_names = list(scores.keys())
    n = len(model_names)
    cols = 3
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=model_names,
                        shared_xaxes=False, shared_yaxes=False)

    for i, name in enumerate(model_names):
        row = i // cols + 1
        col = i % cols + 1
        model_files = sorted(MODELS_DIR.glob(f"{name}_iter*.pkl"))
        if not model_files:
            continue
        model = joblib.load(model_files[-1])
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, colorscale=[[0, DARK], [1, ACCENT]],
                       text=cm, texttemplate="%{text}",
                       showscale=False,
                       hovertemplate="Actual/Predicted: %{z}<extra></extra>"),
            row=row, col=col,
        )

    fig.update_layout(
        title=f"Confusion Matrices — All Models (Iteration {iteration})",
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font_color=TEXT, height=300 * rows,
    )
    out = PLOTS / f"05_confusion_matrices_iter{iteration}.html"
    fig.write_html(str(out))
    print(f"    [✓] {out.name}")


def plot_roc_curves(scores: dict, X_test, y_test, iteration: int):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color=GRID), name="Random"))

    colors = px.colors.qualitative.Plotly
    for i, name in enumerate(scores):
        model_files = sorted(MODELS_DIR.glob(f"{name}_iter*.pkl"))
        if not model_files:
            continue
        model = joblib.load(model_files[-1])
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = scores[name]["roc_auc"]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title=f"ROC Curves — All Models (Iteration {iteration})",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        paper_bgcolor=DARK, plot_bgcolor=DARK, font_color=TEXT,
        legend_bgcolor=CARD, xaxis_gridcolor=GRID, yaxis_gridcolor=GRID,
    )
    out = PLOTS / f"06_roc_curves_iter{iteration}.html"
    fig.write_html(str(out))
    print(f"    [✓] {out.name}")


def plot_pair_sample(iteration: int):
    try:
        df = pd.read_csv(CLEANED / "ml_scaled.csv")
        cols = df.select_dtypes(include=np.number).columns.tolist()
        top5 = cols[:5] + [TARGET_COLUMN] if TARGET_COLUMN in df.columns else cols[:6]
        sample = df[top5].sample(min(500, len(df)), random_state=42)
        fig = px.scatter_matrix(
            sample, dimensions=top5[:-1],
            color=TARGET_COLUMN if TARGET_COLUMN in sample.columns else None,
            color_discrete_sequence=[GREEN, RED],
            title="Pair Plot — Top Features",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor=DARK, font_color=TEXT)
        out = PLOTS / f"07_pair_plot_iter{iteration}.html"
        fig.write_html(str(out))
        print(f"    [✓] {out.name}")
    except Exception as e:
        print(f"    [!] Pair plot skipped: {e}")


def save_report(df_scores: pd.DataFrame, iteration: int):
    REPORTS.mkdir(parents=True, exist_ok=True)
    best_row = df_scores.iloc[0]
    report = {
        "iteration": iteration,
        "best_model": best_row["Model"],
        "best_f1": best_row["F1"],
        "best_roc_auc": best_row["ROC-AUC"],
        "all_scores": df_scores.to_dict(orient="records"),
    }
    out = REPORTS / f"report_iter{iteration}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    [✓] Report: {out.name}")
    return report


def run(iteration: int = None):
    print("=" * 60)
    print("STEP 06 — Evaluate + Visualize")
    print("=" * 60)
    PLOTS.mkdir(parents=True, exist_ok=True)

    if iteration is None:
        iteration = latest_iteration()

    scores = load_scores(iteration)
    X_test, y_test = load_test_data()

    print(f"\n[→] Generating visualizations for iteration {iteration}...")
    df_scores = plot_model_comparison(scores, iteration)
    plot_confusion_matrices(scores, X_test, y_test, iteration)
    plot_roc_curves(scores, X_test, y_test, iteration)
    plot_pair_sample(iteration)
    report = save_report(df_scores, iteration)

    print(f"\n[✓] Step 06 complete")
    print(f"    Best: {report['best_model']} — F1={report['best_f1']}  ROC={report['best_roc_auc']}")
    return report


if __name__ == "__main__":
    run()
