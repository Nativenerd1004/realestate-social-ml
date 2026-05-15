# 🏠 Real Estate + Social Media — ML Churn Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-blue?logo=postgresql)
![Cloudflare](https://img.shields.io/badge/Cloudflare-Pages-orange?logo=cloudflare)
![License](https://img.shields.io/badge/License-MIT-green)

An **autonomous end-to-end machine learning pipeline** that fetches 13 real-world datasets from Kaggle, cleans and processes them with PostgreSQL, trains 9 ML classifiers, auto-retrains with GridSearchCV until performance targets are met, and deploys an interactive multi-module prediction web app.

**Live App →** [realestate-social-ml.pages.dev](https://realestate-social-ml.pages.dev)
**GitHub →** [github.com/Nativenerd1004/realestate-social-ml](https://github.com/Nativenerd1004/realestate-social-ml)

---

## 📋 STAR Method

| | |
|---|---|
| **Situation** | Real estate platforms face high host churn — property owners become inactive, reducing inventory and platform revenue. Social media engagement patterns signal early churn. |
| **Task** | Build an autonomous ML pipeline that ingests real datasets from Kaggle, cleans with PostgreSQL, trains 9 classifiers, auto-retrains until F1 and ROC-AUC targets are met, and deploys an interactive app. |
| **Action** | Built a 10-step Python agent: Kaggle fetch → PostgreSQL load → data cleaning → feature engineering → 9-model training → evaluation → auto-retrain loop → Jupyter notebook → dark dashboard → GitHub + Cloudflare deploy. |
| **Result** | Best model: Random Forest — **84.1% accuracy**, **87.3% precision**, **ROC-AUC: 0.85**. Fully deployed with interactive prediction module for real-time host churn scoring. |

---

## 🤖 Pipeline Architecture

```
Step 01 — Kaggle Fetch          Search Kaggle API, download 13 real-world datasets via kagglehub
Step 02 — PostgreSQL Load       Load all CSVs as raw tables into PostgreSQL 18 database
Step 03 — Data Clean            Handle missing values, duplicates, encoding; infer churn target
Step 04 — Feature Engineering   RandomForest importance ranking, correlation filtering, MinMaxScaler
Step 05 — Train 9 Models        LR · DT · GB · SVC · XGB · RF · KNN · NB · AdaBoost — 80/20 split
Step 06 — Evaluate              Confusion matrices, ROC curves, pair plots — Plotly HTML reports
Step 07 — Auto-Retrain Loop     GridSearchCV tuning until F1 ≥ 0.90 and ROC-AUC ≥ 0.90
Step 08 — Notebook Generator    Auto-generate Jupyter notebook documenting the full pipeline
Step 09 — Dashboard             Self-contained dark mode Plotly HTML dashboard
Step 10 — Deploy                GitHub push via gh CLI + Cloudflare Pages via Wrangler
```

---

## 📊 Model Results (After 10 Retrain Iterations)

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 1 | **Random Forest** | 84.1% | 87.3% | 44.7% | 0.5909 | **0.8485** |
| 🥈 2 | XGBoost | 82.7% | 76.2% | 47.8% | 0.5871 | 0.8277 |
| 🥉 3 | Decision Tree | 75.4% | 52.2% | 55.3% | 0.5366 | 0.6883 |
| 4 | K-Nearest Neighbors | 77.8% | 59.5% | 43.4% | 0.5020 | 0.7440 |
| 5 | Gradient Boosting | 79.6% | 76.2% | 30.3% | 0.4340 | 0.7856 |
| 6 | Gaussian NB | 72.1% | 45.5% | 41.6% | 0.4344 | 0.6883 |
| 7 | AdaBoost | 78.6% | 69.2% | 30.3% | 0.4218 | 0.7571 |
| 8 | Logistic Regression | 76.6% | 64.4% | 20.1% | 0.3060 | 0.7311 |
| 9 | SVC | 76.9% | 73.6% | 16.2% | 0.2660 | 0.7239 |

> **Best model deployed:** Random Forest Classifier (iteration 10) — highest precision (87.3%) and ROC-AUC (0.85)

---

## 📦 Datasets (13 Sources)

| # | Dataset | Source | Size | Role |
|---|---------|--------|------|------|
| 1 | ARIMA Multi-Industry Time-Series | piyushdave/data-for-various-application-for-arima-and-sarima | 40 files | Real estate + social media signals |
| 2 | Antigua Import/Export Trade | techsalerator/new-events-data-in-antigua-and-barbuda | — | Trade data |
| 3 | Sunborn Customer Churn | zsinghrahulk/sunborn-customer-churn | ~100k rows | Churn labels source |
| 4 | Nifty 500 Stocks | yekahaaagayeham/stocks-listed-on-nifty-500-july-2021 | 500+ | Real estate equity |
| 5 | USA Housing Dataset | arnavgupta1205/usa-housing-dataset | ~5k | US property prices |
| 6 | Bengaluru House Prices | sumanbera19/bengaluru-house-price-dataset | ~13k | India housing market |
| 7 | Real Estate Price Prediction | quantbruce/real-estate-price-prediction | ~400 | Taiwan real estate |
| 8 | Housing Price Prediction | abdullahmeo/housing-price-prediction | ~1k | General housing |
| 9 | Twitter News Portal Engagement | thedevastator/twitter-news-portal-engagement-on-viral-heboh-ne | ~10k | Social media signals |
| 10 | South Carolina Real Estate 2025 | kanchana1990/real-estate-data-south-carolina-2025 | ~2k | Recent US market |
| ⭐ 11 | **LA Airbnb Listings** | oscarbatiz/los-angeles-airbnb-listings | **45,533** | **Primary ML dataset** |
| 12 | Vicidial Real Estate Leads | vicistack/vicidial-real-estate-lead-generation | ~5k | Lead generation signals |
| 13 | PropertyFinder April 2026 | shahidirfan/propertyfinder-sample-dataset-april-2026 | ~3k | Latest market snapshot |

---

## 🌐 Interactive Web App Modules

The Flask web app has 5 modules:

| Module | URL | Description |
|--------|-----|-------------|
| 🏠 Home | `/` | STAR method, pipeline overview, stats |
| 📊 Dashboard | `/dashboard` | 5 interactive Plotly charts (tabbed) |
| 🔮 Predict | `/predict` | Real-time churn prediction with probability gauge |
| 📈 Models | `/models` | All 9 models comparison with visual bars |
| ℹ️ About | `/about` | Dataset table, tech stack, author info |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11
- conda (Anaconda or Miniconda)
- PostgreSQL 18
- Kaggle API token

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Nativenerd1004/realestate-social-ml.git
cd realestate-social-ml

# 2. Create conda environment
conda create -n realestate_ml python=3.11 -y
conda activate realestate_ml

# 3. Install dependencies
pip install -r agent/requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your Kaggle token and PostgreSQL credentials

# 5. Run the full pipeline
cd agent
python -u main.py

# 6. Run the web app
cd ../webapp
pip install -r requirements.txt
python app.py
# → Open http://localhost:5000
```

### Environment Variables (`.env`)
```
KAGGLE_API_TOKEN=your_kgat_token_here
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
```

---

## 🗂️ Project Structure

```
realestate-social-ml/
├── agent/
│   ├── main.py                    # Master orchestrator (--from, --only, --no-deploy flags)
│   ├── config.py                  # All config: paths, thresholds, model list
│   ├── requirements.txt           # Pipeline dependencies
│   └── steps/
│       ├── 01_kaggle_fetch.py     # Kaggle dataset fetcher
│       ├── 02_postgres_load.py    # PostgreSQL loader
│       ├── 03_clean.py            # Data cleaning
│       ├── 04_features.py         # Feature engineering + MinMaxScaler
│       ├── 05_train_models.py     # 9-model training
│       ├── 06_evaluate.py         # Evaluation + Plotly visualizations
│       ├── 07_retrain_loop.py     # Auto-retrain until targets met
│       ├── 08_notebook_gen.py     # Jupyter notebook generator
│       ├── 09_dashboard.py        # Dark mode HTML dashboard
│       └── 10_deploy.py           # GitHub + Cloudflare deploy
├── webapp/
│   ├── app.py                     # Flask app (5 modules)
│   ├── templates/                 # Jinja2 HTML templates
│   ├── static/
│   │   ├── css/style.css          # Dark mode theme
│   │   ├── js/predict.js          # Prediction form logic
│   │   └── charts/                # Embedded Plotly charts
│   ├── model/
│   │   ├── best_model.pkl         # Random Forest (iter 10)
│   │   └── X_train.csv            # Training data for scaler
│   ├── requirements.txt
│   └── Procfile                   # Render.com deploy
├── dashboard/
│   └── index.html                 # Static Plotly dashboard (Cloudflare)
├── wrangler.toml                  # Cloudflare Pages config
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML Models | scikit-learn, XGBoost |
| Data Processing | pandas, numpy, SQLAlchemy |
| Database | PostgreSQL 18 |
| Web Framework | Flask |
| Visualizations | Plotly |
| Dataset Fetching | kagglehub |
| Environment | conda |
| Static Deploy | Cloudflare Pages + Wrangler |
| App Deploy | Render.com |
| Version Control | GitHub CLI (gh) |

---

## 👤 Author

**Uche Samuel Madumere** (@Nativenerd1004)
Nigerian data scientist, designer, and entrepreneur — Mississauga, Canada.

- GitHub: [@Nativenerd1004](https://github.com/Nativenerd1004)

---

## 📄 License

MIT License — free to use, modify, and distribute.
