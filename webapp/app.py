"""
Real Estate + Social Media ML — Interactive Web App
Flask backend: Home | Dashboard | Predict | Models | About
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "best_model.pkl"
DATA_PATH  = BASE / "model" / "X_train.csv"

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

# ---------------------------------------------------------------------------
# Load model + fit scaler once at startup
# ---------------------------------------------------------------------------
model  = joblib.load(MODEL_PATH)
X_ref  = pd.read_csv(DATA_PATH)
FEATURE_COLS = list(X_ref.columns)

# Re-fit scaler on the raw (already-encoded) training data
scaler = MinMaxScaler()
scaler.fit(X_ref)

# Medians for non-user-facing features
MEDIANS = X_ref.median().to_dict()

# Model results (final iteration)
MODEL_RESULTS = [
    {"model": "Random Forest",        "accuracy": 0.8407, "precision": 0.8733, "recall": 0.4465, "f1": 0.5909, "roc_auc": 0.8485},
    {"model": "XGBoost",              "accuracy": 0.8268, "precision": 0.7615, "recall": 0.4776, "f1": 0.5871, "roc_auc": 0.8277},
    {"model": "Decision Tree",        "accuracy": 0.7540, "precision": 0.5215, "recall": 0.5526, "f1": 0.5366, "roc_auc": 0.6883},
    {"model": "K-Nearest Neighbors",  "accuracy": 0.7780, "precision": 0.5949, "recall": 0.4342, "f1": 0.5020, "roc_auc": 0.7440},
    {"model": "Gradient Boosting",    "accuracy": 0.7961, "precision": 0.7623, "recall": 0.3034, "f1": 0.4340, "roc_auc": 0.7856},
    {"model": "Gaussian NB",          "accuracy": 0.7209, "precision": 0.4546, "recall": 0.4159, "f1": 0.4344, "roc_auc": 0.6883},
    {"model": "AdaBoost",             "accuracy": 0.7857, "precision": 0.6919, "recall": 0.3034, "f1": 0.4218, "roc_auc": 0.7571},
    {"model": "Logistic Regression",  "accuracy": 0.7655, "precision": 0.6443, "recall": 0.2007, "f1": 0.3060, "roc_auc": 0.7311},
    {"model": "SVC",                  "accuracy": 0.7691, "precision": 0.7355, "recall": 0.1623, "f1": 0.2660, "roc_auc": 0.7239},
]

DATASETS = [
    {"name": "ARIMA Time-Series Multi-Industry", "ref": "piyushdave/data-for-various-application-for-arima-and-sarima", "rows": "40 files", "note": "Real estate + social media time-series"},
    {"name": "Antigua Import/Export Trade",       "ref": "techsalerator/new-events-data-in-antigua-and-barbuda",         "rows": "—",       "note": "Trade data"},
    {"name": "Sunborn Customer Churn",            "ref": "zsinghrahulk/sunborn-customer-churn",                           "rows": "~100k",   "note": "Primary churn labels source"},
    {"name": "Nifty 500 Stocks",                  "ref": "yekahaaagayeham/stocks-listed-on-nifty-500-july-2021",          "rows": "500+",    "note": "Indian real estate equity"},
    {"name": "USA Housing Dataset",               "ref": "arnavgupta1205/usa-housing-dataset",                            "rows": "~5k",     "note": "US property prices"},
    {"name": "Bengaluru House Prices",            "ref": "sumanbera19/bengaluru-house-price-dataset",                     "rows": "~13k",    "note": "India housing market"},
    {"name": "Real Estate Price Prediction",      "ref": "quantbruce/real-estate-price-prediction",                       "rows": "~400",    "note": "Taiwan real estate"},
    {"name": "Housing Price Prediction",          "ref": "abdullahmeo/housing-price-prediction",                          "rows": "~1k",     "note": "General housing"},
    {"name": "Twitter News Portal Engagement",    "ref": "thedevastator/twitter-news-portal-engagement-on-viral-heboh-ne","rows": "~10k",   "note": "Social media engagement"},
    {"name": "South Carolina Real Estate 2025",   "ref": "kanchana1990/real-estate-data-south-carolina-2025",             "rows": "~2k",     "note": "Recent US market data"},
    {"name": "LA Airbnb Listings ★ Primary",     "ref": "oscarbatiz/los-angeles-airbnb-listings",                        "rows": "45,533",  "note": "Primary ML dataset — host churn"},
    {"name": "Vicidial Real Estate Leads",        "ref": "vicistack/vicidial-real-estate-lead-generation",                "rows": "~5k",     "note": "Lead generation signals"},
    {"name": "PropertyFinder April 2026",         "ref": "shahidirfan/propertyfinder-sample-dataset-april-2026",          "rows": "~3k",     "note": "Latest market snapshot"},
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        # Build full feature row using medians as base
        row = MEDIANS.copy()

        # Override with user-provided values (already in original scale)
        user_map = {
            "price":              float(data.get("price", 155)),
            "availability_365":   float(data.get("availability_365", 202)),
            "minimum_nights":     float(data.get("minimum_nights", 14)),
            "number_of_reviews":  float(data.get("number_of_reviews", 6)),
            "review_scores_rating": float(data.get("review_scores_rating", 4.9)),
            "host_response_rate": float(data.get("host_response_rate", 1.0)),
            "accommodates":       float(data.get("accommodates", 3)),
            "beds":               float(data.get("beds", 2)),
            "bedrooms":           float(data.get("bedrooms", 1)),
            "bathrooms":          float(data.get("bathrooms", 1)),
            "room_type":          float(data.get("room_type", 0)),
            "host_is_superhost":  float(data.get("host_is_superhost", 0)),
            "host_response_time": float(data.get("host_response_time", 0)),
        }
        row.update(user_map)

        # Build DataFrame in correct column order
        X = pd.DataFrame([row])[FEATURE_COLS]

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        pred  = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])

        label   = "High Churn Risk" if pred == 1 else "Low Churn Risk"
        color   = "#ef4444" if pred == 1 else "#22c55e"
        pct     = round(proba * 100, 1)

        # Key factors (top feature importances)
        importances = model.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:5]
        top_features = [
            {"feature": FEATURE_COLS[i].replace("_", " ").title(), "importance": round(float(importances[i]), 4)}
            for i in top_idx
        ]

        return jsonify({
            "prediction": pred,
            "label": label,
            "color": color,
            "probability": pct,
            "top_features": top_features,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/models")
def models_page():
    return render_template("models.html", results=MODEL_RESULTS)


@app.route("/about")
def about():
    return render_template("about.html", datasets=DATASETS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
