"""
Generate fresh Plotly dashboard charts from the 4 saved ML models.
Outputs 5 self-contained HTML files to webapp/static/charts/.
"""
import json, joblib, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

BASE   = Path("/Users/apple/Desktop/RealEstate_SocialMedia_ML")
MODEL  = BASE / "webapp" / "model"
CHARTS = BASE / "webapp" / "static" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

DARK_BG   = "#0d1117"
SURFACE   = "#161b22"
SURFACE2  = "#21262d"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
ACCENT    = "#58a6ff"

TABS = [
    {"name": "Host Churn",      "prefix": "churn",   "color": "#ef4444"},
    {"name": "Market Position", "prefix": "social",  "color": "#a855f7"},
    {"name": "Listing Sale",    "prefix": "listing", "color": "#22c55e"},
    {"name": "Social Media",    "prefix": "yt",      "color": "#06b6d4"},
]

LAYOUT = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

def save(fig, name):
    path = CHARTS / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved {name}.html")


# ── 1. Model Comparison — F1 and ROC across all 4 tabs ───────────────────────
print("Generating model_comparison.html ...")
all_scores = {}
for t in TABS:
    sc = json.loads((MODEL / f"{t['prefix']}_scores.json").read_text())
    all_scores[t["name"]] = sc

models = list(list(all_scores.values())[0].keys())
tab_names = [t["name"] for t in TABS]
colors = [t["color"] for t in TABS]

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["F1 Score by Model & Tab", "ROC-AUC by Model & Tab"])

for i, t in enumerate(TABS):
    sc = all_scores[t["name"]]
    model_names = list(sc.keys())
    f1s  = [sc[m]["f1"]      for m in model_names]
    rocs = [sc[m]["roc_auc"] for m in model_names]
    fig.add_trace(go.Bar(name=t["name"], x=model_names, y=f1s,
                         marker_color=t["color"], opacity=0.85,
                         legendgroup=t["name"]), row=1, col=1)
    fig.add_trace(go.Bar(name=t["name"], x=model_names, y=rocs,
                         marker_color=t["color"], opacity=0.85,
                         legendgroup=t["name"], showlegend=False), row=1, col=2)

fig.update_layout(**LAYOUT, title="Model Performance Across All 4 Prediction Tabs",
                  barmode="group", legend=dict(bgcolor=SURFACE2, bordercolor=BORDER, borderwidth=1),
                  height=480)
fig.update_xaxes(tickangle=-30, gridcolor=BORDER, zerolinecolor=BORDER)
fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, range=[0, 1])
fig.update_annotations(font_color=TEXT)
save(fig, "model_comparison")


# ── 2. Feature Importance — top 8 features per tab ───────────────────────────
print("Generating feature_importance.html ...")
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=[t["name"] for t in TABS],
                    horizontal_spacing=0.12, vertical_spacing=0.18)

positions = [(1,1),(1,2),(2,1),(2,2)]
for idx, t in enumerate(TABS):
    model = joblib.load(MODEL / f"{t['prefix']}_model.pkl")
    feats = json.loads((MODEL / f"{t['prefix']}_features.json").read_text())
    row, col = positions[idx]
    try:
        imps = model.feature_importances_
        pairs = sorted(zip(feats, imps), key=lambda x: x[1])[-8:]
        names = [p[0].replace("_", " ").title() for p in pairs]
        vals  = [p[1] for p in pairs]
        fig.add_trace(go.Bar(x=vals, y=names, orientation="h",
                             marker_color=t["color"], opacity=0.85,
                             name=t["name"], showlegend=False), row=row, col=col)
    except AttributeError:
        fig.add_trace(go.Bar(x=[0], y=["N/A"], orientation="h",
                             marker_color=t["color"], showlegend=False), row=row, col=col)

fig.update_layout(**LAYOUT, title="Feature Importance — Best Model per Tab", height=620)
fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
fig.update_annotations(font_color=TEXT)
save(fig, "feature_importance")


# ── 3. Target Distribution — class balance per tab ───────────────────────────
print("Generating target_distribution.html ...")
target_data = [
    {"tab": "Host Churn",      "Positive (Churned)": 50.0, "Negative (Active)": 50.0,   "note": "SMOTE balanced from 2.85%"},
    {"tab": "Market Position", "Positive (Above Median)": 49.8, "Negative (Below Median)": 50.2, "note": "235,695 rows · 63 locations"},
    {"tab": "Listing Sale",    "Positive (At/Above Asking)": 59.0, "Negative (Below Asking)": 41.0,  "note": "8,574 Illinois 2026 sales"},
    {"tab": "Social Media",    "Positive (High Engagement)": 50.0, "Negative (Low Engagement)": 50.0, "note": "178,399 YouTube trending videos"},
]

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=[d["tab"] + f"<br><sup>{d['note']}</sup>" for d in target_data],
                    specs=[[{"type":"pie"},{"type":"pie"}],[{"type":"pie"},{"type":"pie"}]],
                    horizontal_spacing=0.05, vertical_spacing=0.12)

for idx, (d, t) in enumerate(zip(target_data, TABS)):
    row, col = positions[idx]
    pos_key = [k for k in d if k not in ("tab","note") and "Positive" in k][0]
    neg_key = [k for k in d if k not in ("tab","note") and "Negative" in k][0]
    fig.add_trace(go.Pie(
        labels=[pos_key, neg_key],
        values=[d[pos_key], d[neg_key]],
        marker_colors=[t["color"], SURFACE2],
        hole=0.55,
        textfont_color=TEXT,
        name=d["tab"],
        showlegend=False,
    ), row=row, col=col)

fig.update_layout(**LAYOUT, title="Target Class Distribution per Prediction Tab", height=560)
fig.update_annotations(font_color=TEXT, font_size=12)
save(fig, "target_distribution")


# ── 4. ROC-AUC Summary — radar / grouped bar ─────────────────────────────────
print("Generating roc_curves.html ...")
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

fig = go.Figure()
for t in TABS:
    sc = json.loads((MODEL / f"{t['prefix']}_scores.json").read_text())
    best_name = json.loads((MODEL / f"{t['prefix']}_best.json").read_text())["model"]
    best = sc[best_name]
    values = [best[m] for m in metrics] + [best[metrics[0]]]
    labels = metric_labels + [metric_labels[0]]
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels,
        fill="toself", opacity=0.55,
        name=f"{t['name']} ({best_name})",
        line_color=t["color"],
        marker_color=t["color"],
    ))

fig.update_layout(
    **LAYOUT,
    title="Best Model Metrics — Radar Comparison Across 4 Tabs",
    polar=dict(
        bgcolor=SURFACE,
        radialaxis=dict(visible=True, range=[0, 1], color=MUTED, gridcolor=BORDER),
        angularaxis=dict(color=TEXT, gridcolor=BORDER),
    ),
    legend=dict(bgcolor=SURFACE2, bordercolor=BORDER, borderwidth=1),
    height=520,
)
save(fig, "roc_curves")


# ── 5. Dataset Overview — row counts ─────────────────────────────────────────
print("Generating correlation_heatmap.html ...")
datasets = [
    {"name": "Real Estate CRM", "tab": "Tab 1 — Host Churn", "rows": 285000, "color": "#ef4444"},
    {"name": "USA Real Estate", "tab": "Tab 2 — Market Pos.", "rows": 1471301, "color": "#a855f7"},
    {"name": "Canadian Houses", "tab": "Tab 2 — Market Pos.", "rows": 35768, "color": "#c084fc"},
    {"name": "Illinois Sales 2026", "tab": "Tab 3 — Listing Sale", "rows": 8574, "color": "#22c55e"},
    {"name": "YouTube Trending", "tab": "Tab 4 — Social Media", "rows": 178399, "color": "#06b6d4"},
]
df_ds = pd.DataFrame(datasets)

fig = go.Figure(go.Bar(
    x=df_ds["name"],
    y=df_ds["rows"],
    marker_color=df_ds["color"],
    text=[f"{r:,}" for r in df_ds["rows"]],
    textposition="outside",
    textfont_color=TEXT,
    customdata=df_ds["tab"],
    hovertemplate="<b>%{x}</b><br>Rows: %{y:,}<br>%{customdata}<extra></extra>",
))
fig.update_layout(
    **LAYOUT,
    title="Training Dataset Sizes — Rows per Dataset",
    xaxis_title="Dataset",
    yaxis_title="Row Count",
    yaxis_type="log",
    height=460,
)
save(fig, "correlation_heatmap")

print("\nAll 5 charts generated.")
