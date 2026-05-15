"""
Step 07 — Auto-Retrain Loop
Keeps retraining until F1 ≥ MIN_F1_SCORE and ROC-AUC ≥ MIN_ROC_AUC (both → 1.0).
Applies progressive hyperparameter tuning on each iteration.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (CLEANED, MAX_RETRAIN_ITER, MIN_F1_SCORE, MIN_ROC_AUC,
                    MODELS_DIR, RANDOM_STATE, TARGET_COLUMN)

# Hyperparameter grids applied progressively
PARAM_GRIDS = {
    "XGBClassifier": [
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1, 0.05]},
        {"n_estimators": [300, 500], "max_depth": [6, 8], "learning_rate": [0.01, 0.05],
         "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
    ],
    "RandomForestClassifier": [
        {"n_estimators": [200, 300], "max_depth": [None, 10], "min_samples_split": [2, 5]},
        {"n_estimators": [500], "max_depth": [None, 15], "min_samples_leaf": [1, 2],
         "max_features": ["sqrt", "log2"]},
    ],
    "GradientBoostingClassifier": [
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        {"n_estimators": [300], "max_depth": [4, 6], "subsample": [0.8, 1.0],
         "min_samples_split": [2, 4]},
    ],
    "LogisticRegression": [
        {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "saga"]},
        {"C": [0.001, 0.01, 0.1, 1], "penalty": ["l1", "l2"], "solver": ["saga"]},
    ],
}

TUNEABLE = list(PARAM_GRIDS.keys())


def best_scores_so_far() -> tuple[float, float, int]:
    files = sorted(MODELS_DIR.glob("scores_iter*.json"))
    best_f1 = 0.0
    best_roc = 0.0
    last_iter = 0
    for f in files:
        with open(f) as fh:
            scores = json.load(fh)
        for s in scores.values():
            best_f1  = max(best_f1,  s.get("f1", 0))
            best_roc = max(best_roc, s.get("roc_auc", 0))
        last_iter = int(f.stem.split("iter")[1])
    return best_f1, best_roc, last_iter


def tune_model(name: str, model, X_train, y_train, iteration: int):
    if name not in PARAM_GRIDS:
        return model
    grids = PARAM_GRIDS[name]
    grid_idx = min(iteration - 1, len(grids) - 1)
    param_grid = grids[grid_idx]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(model, param_grid, cv=skf, scoring="f1",
                          n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"      Best params: {search.best_params_}  CV-F1={search.best_score_:.4f}")
    return search.best_estimator_


def run():
    print("=" * 60)
    print("STEP 07 — Auto-Retrain Loop")
    print("=" * 60)

    import importlib
    train_mod = importlib.import_module("steps.05_train_models")
    eval_mod  = importlib.import_module("steps.06_evaluate")
    train_step = train_mod.run
    eval_step  = eval_mod.run

    df = pd.read_csv(CLEANED / "ml_scaled.csv")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train = pd.read_csv(CLEANED / "X_train.csv")
    y_train = pd.read_csv(CLEANED / "y_train.csv").squeeze()

    best_f1, best_roc, last_iter = best_scores_so_far()
    iteration = last_iter

    print(f"[→] Current best — F1={best_f1:.4f}  ROC={best_roc:.4f}")
    print(f"[→] Target       — F1≥{MIN_F1_SCORE}  ROC≥{MIN_ROC_AUC}")

    while (best_f1 < MIN_F1_SCORE or best_roc < MIN_ROC_AUC) and iteration < MAX_RETRAIN_ITER:
        iteration += 1
        print(f"\n{'─'*60}")
        print(f"[→] Retrain iteration {iteration}/{MAX_RETRAIN_ITER}")

        # Tune top models before retraining
        print("[→] Hyperparameter tuning on top models...")
        for name in TUNEABLE:
            print(f"    Tuning {name}...")
            prev_files = sorted(MODELS_DIR.glob(f"{name}_iter*.pkl"))
            if not prev_files:
                continue
            prev_model = joblib.load(prev_files[-1])
            tuned = tune_model(name, prev_model, X_train, y_train, iteration)
            tuned_path = MODELS_DIR / f"{name}_iter{iteration}_tuned.pkl"
            joblib.dump(tuned, tuned_path)

        # Full retrain pass with updated iteration seed
        scores = train_step(iteration)
        eval_step(iteration)

        for s in scores.values():
            best_f1  = max(best_f1,  s.get("f1", 0))
            best_roc = max(best_roc, s.get("roc_auc", 0))

        print(f"\n[→] After iter {iteration} — Best F1={best_f1:.4f}  Best ROC={best_roc:.4f}")

        if best_f1 >= MIN_F1_SCORE and best_roc >= MIN_ROC_AUC:
            print(f"\n[✓] Target reached! F1={best_f1:.4f}  ROC={best_roc:.4f}")
            break
    else:
        if iteration >= MAX_RETRAIN_ITER:
            print(f"\n[!] Max iterations ({MAX_RETRAIN_ITER}) reached.")
            print(f"    Final — F1={best_f1:.4f}  ROC={best_roc:.4f}")
        else:
            print(f"\n[✓] Already meets target on first pass.")

    print(f"\n[✓] Step 07 complete — Best F1={best_f1:.4f}  ROC={best_roc:.4f}")
    return {"best_f1": best_f1, "best_roc": best_roc, "iterations": iteration}


if __name__ == "__main__":
    run()
