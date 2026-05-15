"""
Step 05 — Train All 9 ML Models (Samuel's Pipeline)
Trains every model, saves them, and returns scores for the retrain loop.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (CLEANED, MODELS_DIR, RANDOM_STATE, TARGET_COLUMN,
                    TEST_SIZE, CV_FOLDS)


def build_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVC": SVC(probability=True, random_state=RANDOM_STATE),
        "XGBClassifier": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
        "GaussianNB": GaussianNB(),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=RANDOM_STATE),
    }


def train(iteration: int = 1) -> dict:
    print("=" * 60)
    print(f"STEP 05 — Model Training (Iteration {iteration})")
    print("=" * 60)

    path = CLEANED / "ml_scaled.csv"
    if not path.exists():
        print("[!] ml_scaled.csv not found — run Step 04 first")
        sys.exit(1)

    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + iteration, stratify=y
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = build_models()
    results = {}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n[→] Training: {name}")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc   = accuracy_score(y_test, y_pred)
            prec  = precision_score(y_test, y_pred, zero_division=0)
            rec   = recall_score(y_test, y_pred, zero_division=0)
            f1    = f1_score(y_test, y_pred, zero_division=0)
            roc   = roc_auc_score(y_test, y_prob)
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1).mean()

            results[name] = {
                "accuracy":  round(acc, 4),
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1":        round(f1, 4),
                "roc_auc":   round(roc, 4),
                "cv_f1":     round(cv_f1, 4),
                "iteration": iteration,
            }
            print(f"    Acc={acc:.4f}  F1={f1:.4f}  ROC={roc:.4f}  CV-F1={cv_f1:.4f}")

            # Save model
            model_path = MODELS_DIR / f"{name}_iter{iteration}.pkl"
            joblib.dump(model, model_path)

        except Exception as e:
            print(f"    [!] {name} failed: {e}")

    # Save scores
    scores_path = MODELS_DIR / f"scores_iter{iteration}.json"
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save X_test/y_test for evaluation step
    X_test.to_csv(CLEANED / "X_test.csv", index=False)
    y_test.to_csv(CLEANED / "y_test.csv", index=False)
    X_train.to_csv(CLEANED / "X_train.csv", index=False)
    y_train.to_csv(CLEANED / "y_train.csv", index=False)

    best = max(results, key=lambda k: (results[k]["f1"] + results[k]["roc_auc"]) / 2)
    print(f"\n[✓] Best model: {best} — F1={results[best]['f1']}  ROC={results[best]['roc_auc']}")
    print(f"[✓] Step 05 complete — scores saved to {scores_path}")
    return results


def run(iteration: int = 1):
    return train(iteration)


if __name__ == "__main__":
    run()
