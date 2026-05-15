"""
Step 04 — Feature Engineering & Selection
Generates correlation heatmap, selects top features, scales with MinMaxScaler.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED, PLOTS, TARGET_COLUMN, RANDOM_STATE


DARK = "#0d1117"
GRID = "#30363d"
TEXT = "#c9d1d9"
ACCENT = "#58a6ff"


def load_data():
    path = CLEANED / "ml_ready.csv"
    if not path.exists():
        print("[!] ml_ready.csv not found — run Step 03 first")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[→] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def plot_target_distribution(df: pd.DataFrame):
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = [TARGET_COLUMN, "count"]
    counts[TARGET_COLUMN] = counts[TARGET_COLUMN].map({0: "No Churn", 1: "Churn"})

    fig = px.bar(
        counts, x=TARGET_COLUMN, y="count",
        color=TARGET_COLUMN,
        color_discrete_sequence=["#238636", "#da3633"],
        title="Target Distribution — Churn vs No Churn",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font_color=TEXT, showlegend=False,
    )
    out = PLOTS / "01_target_distribution.html"
    fig.write_html(str(out))
    print(f"    [✓] {out.name}")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    cols = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols,
        colorscale="RdBu", zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Correlation Heatmap",
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font_color=TEXT, template="plotly_dark",
        height=700,
    )
    out = PLOTS / "02_correlation_heatmap.html"
    fig.write_html(str(out))
    print(f"    [✓] {out.name}")
    return corr


def select_features(df: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Drop features with near-zero importance
    selected = importances[importances > 0.001].index.tolist()

    # Also drop highly correlated pairs (|r| > 0.95), keeping the more important one
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        highly_corr = upper.index[upper[col].abs() > 0.95].tolist()
        for hc in highly_corr:
            if importances.get(hc, 0) < importances.get(col, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)
    selected = [f for f in selected if f not in to_drop]

    print(f"[→] Features selected: {len(selected)} / {len(X.columns)}")

    # Plot feature importance
    top20 = importances[selected[:20]].reset_index()
    top20.columns = ["feature", "importance"]
    fig = px.bar(
        top20, x="importance", y="feature", orientation="h",
        title="Top 20 Feature Importances (Random Forest)",
        color="importance", color_continuous_scale="Blues",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font_color=TEXT, yaxis={"categoryorder": "total ascending"},
    )
    out = PLOTS / "03_feature_importance.html"
    fig.write_html(str(out))
    print(f"    [✓] {out.name}")

    return selected


def scale_and_save(df: pd.DataFrame, features: list[str]):
    X = df[features].copy()
    y = df[TARGET_COLUMN]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    final = X_scaled.copy()
    final[TARGET_COLUMN] = y.values

    out = CLEANED / "ml_scaled.csv"
    final.to_csv(out, index=False)

    # Save feature list
    feat_path = CLEANED / "selected_features.json"
    with open(feat_path, "w") as f:
        json.dump(features, f, indent=2)

    print(f"[→] Scaled dataset saved: {out}")
    return str(out)


def run():
    print("=" * 60)
    print("STEP 04 — Feature Engineering & Selection")
    print("=" * 60)
    PLOTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print("\n[→] Generating plots...")
    plot_target_distribution(df)
    corr = plot_correlation_heatmap(df)
    features = select_features(df, corr)
    out = scale_and_save(df, features)

    print(f"\n[✓] Step 04 complete")
    return out


if __name__ == "__main__":
    run()
