"""
Step 09 — Dark Mode Interactive Dashboard
Generates a single self-contained index.html with all charts embedded.
Deploys to Cloudflare Pages via GitHub (no server needed).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score,
                              roc_curve, accuracy_score, precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED, DASHBOARD, MODELS_DIR, PLOTS, REPORTS, TARGET_COLUMN

DARK   = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
TEXT   = "#c9d1d9"
MUTED  = "#8b949e"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
PURPLE = "#d2a8ff"
ORANGE = "#f0883e"

pio.templates["dark_custom"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=DARK, plot_bgcolor=CARD,
        font_color=TEXT,
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
    )
)
pio.templates.default = "dark_custom"


def load_latest() -> tuple[dict, int]:
    files = sorted(MODELS_DIR.glob("scores_iter*.json"))
    if not files:
        return {}, 1
    last = files[-1]
    iteration = int(last.stem.split("iter")[1])
    with open(last) as f:
        return json.load(f), iteration


def chart_html(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": True, "responsive": True})


def build_model_comparison(scores: dict) -> str:
    rows = [{"Model": k, "F1": v["f1"], "ROC-AUC": v["roc_auc"],
             "Accuracy": v["accuracy"], "Precision": v["precision"],
             "Recall": v["recall"]}
            for k, v in scores.items()]
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)

    fig = go.Figure()
    palette = [ACCENT, GREEN, ORANGE, PURPLE, RED]
    for i, metric in enumerate(["F1", "ROC-AUC", "Accuracy", "Precision", "Recall"]):
        fig.add_trace(go.Bar(name=metric, x=df["Model"], y=df[metric],
                             marker_color=palette[i]))
    fig.update_layout(barmode="group", title="Model Performance Comparison",
                      xaxis_tickangle=-30, height=420)
    return chart_html(fig)


def build_roc_all(scores: dict, X_test, y_test) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color=BORDER), name="Random"))
    colors = [ACCENT, GREEN, RED, ORANGE, PURPLE, "#79c0ff", "#56d364", "#ffa657", "#ff7b72", "#d2a8ff"]
    for i, name in enumerate(scores):
        mf = sorted(MODELS_DIR.glob(f"{name}_iter*.pkl"))
        if not mf:
            continue
        model = joblib.load(mf[-1])
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = scores[name]["roc_auc"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
                                 line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(title="ROC Curves — All Models",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", height=450)
    return chart_html(fig)


def build_confusion_grid(scores: dict, X_test, y_test) -> str:
    names = list(scores.keys())
    cols = 3
    rows_n = (len(names) + cols - 1) // cols
    fig = make_subplots(rows=rows_n, cols=cols, subplot_titles=names)
    for i, name in enumerate(names):
        mf = sorted(MODELS_DIR.glob(f"{name}_iter*.pkl"))
        if not mf:
            continue
        model = joblib.load(mf[-1])
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig.add_trace(go.Heatmap(
            z=cm, colorscale=[[0, CARD], [1, ACCENT]],
            text=cm, texttemplate="%{text}", showscale=False,
        ), row=i // cols + 1, col=i % cols + 1)
    fig.update_layout(title="Confusion Matrices", height=320 * rows_n)
    return chart_html(fig)


def build_target_dist() -> str:
    df = pd.read_csv(CLEANED / "ml_ready.csv")
    vc = df[TARGET_COLUMN].value_counts().reset_index()
    vc.columns = [TARGET_COLUMN, "count"]
    vc[TARGET_COLUMN] = vc[TARGET_COLUMN].map({0: "No Churn", 1: "Churn"})
    fig = px.pie(vc, names=TARGET_COLUMN, values="count",
                 color=TARGET_COLUMN, color_discrete_map={"No Churn": GREEN, "Churn": RED},
                 title="Target Distribution", hole=0.4)
    fig.update_layout(height=350)
    return chart_html(fig)


def build_feature_importance() -> str:
    path = PLOTS / "03_feature_importance.html"
    if path.exists():
        content = path.read_text()
        start = content.find('<div id="')
        end   = content.rfind("</div>") + 6
        return content[start:end] if start != -1 else "<p>Chart unavailable</p>"
    return "<p>Feature importance chart not yet generated.</p>"


def build_score_cards(scores: dict) -> str:
    best_f1_name  = max(scores, key=lambda k: scores[k]["f1"])
    best_roc_name = max(scores, key=lambda k: scores[k]["roc_auc"])
    best_f1  = scores[best_f1_name]["f1"]
    best_roc = scores[best_roc_name]["roc_auc"]
    avg_acc  = round(sum(s["accuracy"] for s in scores.values()) / len(scores), 4)

    def card(title, val, sub, color):
        return f"""
        <div class="card">
          <div class="card-label">{title}</div>
          <div class="card-value" style="color:{color}">{val}</div>
          <div class="card-sub">{sub}</div>
        </div>"""

    return (
        card("Best F1 Score", best_f1, best_f1_name, GREEN) +
        card("Best ROC-AUC", best_roc, best_roc_name, ACCENT) +
        card("Avg Accuracy", avg_acc, f"{len(scores)} models", ORANGE) +
        card("Models Trained", len(scores), "All 9 pipeline models", PURPLE)
    )


def generate_html(scores: dict, iteration: int, X_test, y_test) -> str:
    model_cmp  = build_model_comparison(scores)
    roc_chart  = build_roc_all(scores, X_test, y_test)
    cm_grid    = build_confusion_grid(scores, X_test, y_test)
    dist_chart = build_target_dist()
    feat_chart = build_feature_importance()
    cards_html = build_score_cards(scores)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Real Estate + Social Media ML Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: {DARK}; --card: {CARD}; --border: {BORDER};
      --text: {TEXT}; --muted: {MUTED}; --accent: {ACCENT};
      --green: {GREEN}; --red: {RED};
    }}
    body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; min-height: 100vh; }}
    header {{ background: var(--card); border-bottom: 1px solid var(--border); padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }}
    header h1 {{ font-size: 1.2rem; font-weight: 600; color: var(--text); }}
    header .badge {{ background: var(--accent); color: #000; border-radius: 20px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
    h2 {{ font-size: 0.85rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin: 32px 0 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 8px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px 24px; }}
    .card-label {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .card-value {{ font-size: 2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }}
    .card-sub {{ font-size: 0.75rem; color: var(--muted); }}
    .chart-box {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 24px; overflow: hidden; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; padding: 32px; border-top: 1px solid var(--border); margin-top: 40px; }}
  </style>
</head>
<body>

<header>
  <h1>Real Estate + Social Media &mdash; ML Dashboard</h1>
  <span class="badge">Iteration {iteration}</span>
</header>

<div class="container">

  <h2>Overview</h2>
  <div class="cards">{cards_html}</div>

  <h2>Model Performance</h2>
  <div class="chart-box">{model_cmp}</div>

  <div class="grid-2">
    <div>
      <h2>Target Distribution</h2>
      <div class="chart-box">{dist_chart}</div>
    </div>
    <div>
      <h2>Feature Importance</h2>
      <div class="chart-box">{feat_chart}</div>
    </div>
  </div>

  <h2>ROC Curves</h2>
  <div class="chart-box">{roc_chart}</div>

  <h2>Confusion Matrices</h2>
  <div class="chart-box">{cm_grid}</div>

</div>

<footer>
  Built with Samuel's ML Pipeline &mdash; Real Estate + Social Media &mdash; Iteration {iteration}
</footer>
</body>
</html>"""


def run():
    print("=" * 60)
    print("STEP 09 — Dark Mode Dashboard")
    print("=" * 60)

    DASHBOARD.mkdir(parents=True, exist_ok=True)
    scores, iteration = load_latest()

    if not scores:
        print("[!] No scores found — run Steps 05-06 first")
        sys.exit(1)

    X_test = pd.read_csv(CLEANED / "X_test.csv")
    y_test = pd.read_csv(CLEANED / "y_test.csv").squeeze()

    print(f"[→] Building dashboard for iteration {iteration}...")
    html = generate_html(scores, iteration, X_test, y_test)

    out = DASHBOARD / "index.html"
    out.write_text(html)
    print(f"[✓] Dashboard saved: {out}")
    print(f"    Open in browser: file://{out}")
    return str(out)


if __name__ == "__main__":
    run()
