"""
Real Estate + Social Media ML — 4-Tab Prediction App
Tabs: Host Churn | Market Position (USA+Canada) | Listing Sale | Social Media
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE  = Path(__file__).parent
MDL   = BASE / "model"

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)


def load_model_bundle(name):
    """Load model, scaler, features, and median reference for a named bundle."""
    m   = joblib.load(MDL / f"{name}_model.pkl")
    sc  = joblib.load(MDL / f"{name}_scaler.pkl")
    ref = pd.read_csv(MDL / f"{name}_X_ref.csv")
    ft  = json.loads((MDL / f"{name}_features.json").read_text())
    med = ref.median().to_dict()
    return m, sc, ft, med


# ── Load all four models at startup ───────────────────────────────────────────
churn_model,   churn_scaler,   CHURN_FEATURES,   CHURN_MEDS   = load_model_bundle("churn")
social_model,  social_scaler,  SOCIAL_FEATURES,  SOCIAL_MEDS  = load_model_bundle("social")
listing_model, listing_scaler, LISTING_FEATURES, LISTING_MEDS = load_model_bundle("listing")
yt_model,      yt_scaler,      YT_FEATURES,      YT_MEDS      = load_model_bundle("yt")

# ── Encoder label lists ────────────────────────────────────────────────────────
CHURN_ENC   = {
    "segments":     json.loads((MDL / "churn_segments.json").read_text()),
    "channels":     json.loads((MDL / "churn_channels.json").read_text()),
    "income_bands": json.loads((MDL / "churn_income_bands.json").read_text()),
}
SOCIAL_ENC  = {
    "area_types": json.loads((MDL / "social_area_types.json").read_text()),
    "states":     json.loads((MDL / "social_states.json").read_text()),
}
LISTING_ENC = {
    "property_types": json.loads((MDL / "listing_types.json").read_text()),
    "sub_types":      json.loads((MDL / "listing_subtypes.json").read_text()),
}
YT_ENC = {
    "categories": json.loads((MDL / "yt_categories.json").read_text()),
    "countries":  json.loads((MDL / "yt_countries.json").read_text()),
}

# ── Combined model results for /models page ────────────────────────────────────
def _load_scores(name, label):
    sc = json.loads((MDL / f"{name}_scores.json").read_text())
    return [{"tab": label, "model": k, **v} for k, v in sc.items()]

ALL_MODEL_RESULTS = (
    _load_scores("churn",   "Host Churn") +
    _load_scores("social",  "Market Position") +
    _load_scores("listing", "Listing Sale") +
    _load_scores("yt",      "Social Media")
)


def _predict(model, scaler, features, medians, user_map):
    row = medians.copy()
    row.update({k: float(v) for k, v in user_map.items() if k in features})
    X       = pd.DataFrame([row])[features]
    X_sc    = scaler.transform(X)
    pred    = int(model.predict(X_sc)[0])
    proba   = float(model.predict_proba(X_sc)[0][1])
    try:
        imps    = model.feature_importances_
        top_idx = np.argsort(imps)[::-1][:5]
        top_feats = [{"feature": features[i].replace("_", " ").title(),
                      "importance": round(float(imps[i]), 4)} for i in top_idx]
    except AttributeError:
        top_feats = []
    return pred, proba, top_feats


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html",
                           churn_enc=CHURN_ENC,
                           social_enc=SOCIAL_ENC,
                           listing_enc=LISTING_ENC,
                           yt_enc=YT_ENC)


@app.route("/predict/churn", methods=["POST"])
def predict_churn():
    try:
        d = request.get_json()
        pred, proba, feats = _predict(churn_model, churn_scaler, CHURN_FEATURES, CHURN_MEDS, d)
        pct   = round(proba * 100, 1)
        label = "High Churn Risk" if pred == 1 else "Low Churn Risk"
        color = "#ef4444" if pred == 1 else "#22c55e"
        return jsonify({"prediction": pred, "label": label, "color": color,
                        "probability": pct, "top_features": feats})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/social", methods=["POST"])
def predict_social():
    try:
        d = request.get_json()
        pred, proba, feats = _predict(social_model, social_scaler, SOCIAL_FEATURES, SOCIAL_MEDS, d)
        pct   = round(proba * 100, 1)
        label = "Above Market — Property Priced Above State/Province Median" if pred == 1 else "Below Market — Property Priced Below State/Province Median"
        color = "#a855f7" if pred == 1 else "#f59e0b"
        return jsonify({"prediction": pred, "label": label, "color": color,
                        "probability": pct, "top_features": feats})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/listing", methods=["POST"])
def predict_listing():
    try:
        d = request.get_json()
        pred, proba, feats = _predict(listing_model, listing_scaler, LISTING_FEATURES, LISTING_MEDS, d)
        pct   = round(proba * 100, 1)
        label = "Likely to Sell at or Above Asking Price" if pred == 1 else "Likely to Sell Below Asking Price"
        color = "#22c55e" if pred == 1 else "#f59e0b"
        return jsonify({"prediction": pred, "label": label, "color": color,
                        "probability": pct, "top_features": feats})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/yt", methods=["POST"])
def predict_yt():
    try:
        d = request.get_json()
        pred, proba, feats = _predict(yt_model, yt_scaler, YT_FEATURES, YT_MEDS, d)
        pct   = round(proba * 100, 1)
        label = "High Engagement — Content Likely to Drive Above-Average Reach" if pred == 1 else "Low Engagement — Below-Average Performance Expected"
        color = "#06b6d4" if pred == 1 else "#f59e0b"
        return jsonify({"prediction": pred, "label": label, "color": color,
                        "probability": pct, "top_features": feats})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/models")
def models_page():
    return render_template("models.html", results=ALL_MODEL_RESULTS)


@app.route("/about")
def about():
    datasets = [
        {"name": "Real Estate Analytics + Churn",        "ref": "real-estate-analytics-revenue-behavior-and-churn",          "rows": "~285k",     "note": "Tab 1 — Churn labels, listings, transactions"},
        {"name": "USA Real Estate Dataset",               "ref": "ahmedshahriarsakib/usa-real-estate-dataset",                "rows": "1,471,301", "note": "Tab 2 — US property market position (above/below state median)"},
        {"name": "Canadian House Prices — Top 45 Cities", "ref": "jeremylarcher/canadian-house-prices-for-top-cities",        "rows": "35,768",    "note": "Tab 2 — Canadian property market position (9 provinces)"},
        {"name": "Illinois Real Estate Sold 2026",        "ref": "kanchana1990/illinois-real-estate-sold-properties-data-2026","rows": "8,574",    "note": "Tab 3 — Sold-to-list ratio prediction"},
        {"name": "YouTube Trending Videos Stats 2026",    "ref": "bsthere/youtube-trending-videos-stats-2026",                "rows": "178,399",   "note": "Tab 4 — Social media content engagement prediction"},
    ]
    return render_template("about.html", datasets=datasets)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
