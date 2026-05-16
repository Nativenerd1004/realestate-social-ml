"""
Real Estate + Social Media ML — 3-Tab Prediction App
Tabs: Host Churn | Social Media Performance | Listing Sale Predictor
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
    """Load model, scaler, features, and optional encoder lists for a named bundle."""
    m   = joblib.load(MDL / f"{name}_model.pkl")
    sc  = joblib.load(MDL / f"{name}_scaler.pkl")
    ref = pd.read_csv(MDL / f"{name}_X_ref.csv")
    ft  = json.loads((MDL / f"{name}_features.json").read_text())
    med = ref.median().to_dict()
    return m, sc, ft, med


# ── Load all three models at startup ──────────────────────────────────────────
churn_model,   churn_scaler,   CHURN_FEATURES,   CHURN_MEDS   = load_model_bundle("churn")
social_model,  social_scaler,  SOCIAL_FEATURES,  SOCIAL_MEDS  = load_model_bundle("social")
listing_model, listing_scaler, LISTING_FEATURES, LISTING_MEDS = load_model_bundle("listing")

# ── Encoder label lists ────────────────────────────────────────────────────────
CHURN_ENC   = {
    "segments":     json.loads((MDL / "churn_segments.json").read_text()),
    "channels":     json.loads((MDL / "churn_channels.json").read_text()),
    "income_bands": json.loads((MDL / "churn_income_bands.json").read_text()),
}
SOCIAL_ENC  = {
    "account_types":   json.loads((MDL / "social_account_types.json").read_text()),
    "media_types":     json.loads((MDL / "social_media_types.json").read_text()),
    "content_cats":    json.loads((MDL / "social_content_cats.json").read_text()),
    "traffic_sources": json.loads((MDL / "social_traffic_sources.json").read_text()),
    "days":            json.loads((MDL / "social_days.json").read_text()),
}
LISTING_ENC = {
    "property_types": json.loads((MDL / "listing_property_types.json").read_text()),
    "list_channels":  json.loads((MDL / "listing_list_channels.json").read_text()),
    "cities":         json.loads((MDL / "listing_cities.json").read_text()),
}

# ── Combined model results for /models page ────────────────────────────────────
def _load_scores(name, label):
    sc = json.loads((MDL / f"{name}_scores.json").read_text())
    return [{"tab": label, "model": k, **v} for k, v in sc.items()]

ALL_MODEL_RESULTS = (
    _load_scores("churn",   "Host Churn") +
    _load_scores("social",  "Social Media") +
    _load_scores("listing", "Listing Sale")
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
                           listing_enc=LISTING_ENC)


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
        label = "High Performance — Likely to Drive Leads" if pred == 1 else "Low Performance — Low Lead Potential"
        color = "#22c55e" if pred == 1 else "#f59e0b"
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
        label = "Likely to Sell" if pred == 1 else "At Risk of Not Selling"
        color = "#22c55e" if pred == 1 else "#ef4444"
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
        {"name": "Real Estate Analytics + Churn",   "ref": "real-estate-analytics-revenue-behavior-and-churn", "rows": "~285k",  "note": "Primary — churn labels, listings, transactions"},
        {"name": "US Airbnb Open Data 2020/2023",    "ref": "us-airbnb-open-data",                              "rows": "459,667","note": "Host behaviour over time"},
        {"name": "Instagram Analytics Dataset",      "ref": "instagram-analytics-dataset",                      "rows": "30,000", "note": "Social media content performance"},
        {"name": "Social Media Performance",         "ref": "social-media-performance-and-engagement-data",     "rows": "10,000", "note": "Cross-platform engagement signals"},
        {"name": "Georgia Real Estate 2026",         "ref": "georgia-real-estate-rentals-intelligence-2026",    "rows": "~5k",    "note": "Recent rental market data"},
        {"name": "Sydney Airbnb Short-Term Rentals", "ref": "airbnb-short-term-rental-data-sydney",             "rows": "~10k",   "note": "International rental benchmark"},
    ]
    return render_template("about.html", datasets=datasets)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
