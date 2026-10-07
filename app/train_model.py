"""Train the diabetes risk model.

Run from the repo root:
    python -m app.train_model

Steps:
1. Load the Pima Indians Diabetes dataset (768 patients, 8 features).
2. Treat impossible zeros (e.g. BMI = 0) as missing values.
3. Hold out 20% as a test set (stratified, so both sets have the same share
   of diabetic patients).
4. Compare candidate models with 5-fold cross-validation on the training set.
5. Fit the chosen model on the training set, score it once on the test set,
   and save the model plus metrics.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features import FEATURES, MODERATE_FROM, ZERO_MEANS_MISSING

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pima-indians-diabetes.csv"
MODEL_PATH = ROOT / "app" / "model.joblib"
METRICS_PATH = ROOT / "app" / "metrics.json"
SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH, names=FEATURES + ["Outcome"])
    X = df[FEATURES].copy()
    X[ZERO_MEANS_MISSING] = X[ZERO_MEANS_MISSING].replace(0, np.nan)
    return X, df["Outcome"], df


def candidates():
    return {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED),
        ),
    }


def scores(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "f1": round(f1_score(y_true, pred), 3),
    }


def main():
    X, y, raw = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Data: {len(X)} patients, {y.mean():.0%} diabetic. "
          f"Train {len(X_train)}, test {len(X_test)}.\n")

    # 1. Compare models with cross-validation (training set only).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    print("5-fold cross-validation (training set):")
    for name, model in candidates().items():
        r = cross_validate(model, X_train, y_train, cv=cv, scoring=["roc_auc", "accuracy"])
        cv_results[name] = {
            "roc_auc": round(r["test_roc_auc"].mean(), 3),
            "roc_auc_std": round(r["test_roc_auc"].std(), 3),
            "accuracy": round(r["test_accuracy"].mean(), 3),
        }
        print(f"  {name:<20} ROC AUC {cv_results[name]['roc_auc']:.3f} "
              f"(+/- {cv_results[name]['roc_auc_std']:.3f})")

    # 2. Logistic regression scores as well as the random forest here, and its
    #    coefficients let the API explain each prediction, so it is the pick.
    chosen = "Logistic regression"
    model = candidates()[chosen].fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # 3. Baseline: the original version of this project (random forest on the
    #    raw data, zeros left in), scored on the same test split.
    raw_train, raw_test = raw.loc[X_train.index, FEATURES], raw.loc[X_test.index, FEATURES]
    baseline = RandomForestClassifier(n_estimators=100, random_state=SEED)
    baseline.fit(raw_train, y_train)
    baseline_proba = baseline.predict_proba(raw_test)[:, 1]

    metrics = {
        "model": chosen,
        "dataset": {"patients": len(X), "diabetic_share": round(y.mean(), 3),
                    "train": len(X_train), "test": len(X_test)},
        "cross_validation": cv_results,
        "test": {
            "at_0.5": scores(y_test, proba, 0.5),
            f"flag_moderate_or_high_at_{MODERATE_FROM}": scores(y_test, proba, MODERATE_FROM),
        },
        "baseline_original_version_test_at_0.5": scores(y_test, baseline_proba, 0.5),
        "coefficients": dict(zip(FEATURES, model[-1].coef_[0].round(3).tolist())),
        "sklearn_version": sklearn.__version__,
    }

    print(f"\nChosen: {chosen}. Held-out test set:")
    for k, v in metrics["test"].items():
        print(f"  {k:<32} {v}")
    print(f"  {'original version (baseline)':<32} {metrics['baseline_original_version_test_at_0.5']}")

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
