"""Train, compare and save loan approval classifiers.

    python -m loan_model.train --data data/loans.csv

Compares Logistic Regression, Random Forest and Gradient Boosting with stratified
5-fold cross-validation, refits the best one, evaluates it on a held-out test set,
computes permutation feature importance, and saves everything to models/.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from loan_model.features import build_preprocessor, split_xy

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                    random_state=42),
}


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def train(data_path: str, out_dir: str = "models", seed: int = 42) -> dict:
    df = pd.read_csv(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(make_pipeline(model), X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = {"mean_roc_auc": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        print(f"{name:<22} CV ROC-AUC {scores.mean():.3f} ± {scores.std():.3f}")

    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean_roc_auc"])
    best = make_pipeline(CANDIDATES[best_name]).fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, pred, target_names=["Rejected", "Approved"]))

    # Permutation importance on raw input columns: how much the test ROC-AUC drops
    # when each column is shuffled. Model-agnostic and easy to explain.
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed)
    importance = (pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    print("Feature importance (drop in ROC-AUC when shuffled):")
    print(importance.head(8).to_string(index=False, float_format="%.4f"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out / "loan_model.joblib")
    importance.to_csv(out / "feature_importance.csv", index=False)
    report = {
        "best_model": best_name,
        "cv": cv_scores,
        "test": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "approval_rate": round(float(y.mean()), 4),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model and reports to {out}/")
    return report


def main():
    p = argparse.ArgumentParser(description="Train the loan approval model.")
    p.add_argument("--data", default="data/loans.csv")
    p.add_argument("--out", default="models")
    args = p.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
