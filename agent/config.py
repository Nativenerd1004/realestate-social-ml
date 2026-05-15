import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/apple/Desktop/RealEstate_SocialMedia_ML")
DATASETS    = BASE_DIR / "datasets"
CLEANED     = BASE_DIR / "cleaned"
MODELS_DIR  = BASE_DIR / "models"
PLOTS       = BASE_DIR / "plots"
NOTEBOOKS   = BASE_DIR / "notebooks"
REPORTS     = BASE_DIR / "reports"
DASHBOARD   = BASE_DIR / "dashboard"

# ── Kaggle ─────────────────────────────────────────────────────────────────────
KAGGLE_SEARCH_QUERIES = [
    "real estate social media marketing",
    "real estate customer churn",
    "housing market prediction",
    "real estate price prediction",
    "social media engagement real estate",
    "property listing performance",
    "real estate lead generation",
]
MAX_DATASETS_PER_QUERY = 2   # download top 2 per query → ~14 datasets total

# ── PostgreSQL ─────────────────────────────────────────────────────────────────
PG_HOST     = "localhost"
PG_PORT     = 5432
PG_USER     = "postgres"
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")   # set PG_PASSWORD env var or update this
PG_DB       = "realestate_ml"
PG_URL      = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# PostgreSQL 18 binary path (EnterpriseDB installer on this machine)
PG_BIN      = "/Library/PostgreSQL/18/bin"

# ── ML Pipeline ───────────────────────────────────────────────────────────────
TARGET_COLUMN     = "churn"          # agent infers this if not found
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
CV_FOLDS          = 5

# Retrain loop thresholds
MIN_F1_SCORE      = 0.90
MIN_ROC_AUC       = 0.90
MAX_RETRAIN_ITER  = 10

# ── Models to train ───────────────────────────────────────────────────────────
MODELS = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "SVC",
    "XGBClassifier",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
    "AdaBoostClassifier",
]

# ── Deployment ────────────────────────────────────────────────────────────────
GITHUB_REPO_NAME  = "realestate-social-ml"
CF_PROJECT_NAME   = "realestate-social-ml"
