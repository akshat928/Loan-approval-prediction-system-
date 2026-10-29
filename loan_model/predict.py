"""Load the trained model and score new applications."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "loan_model.joblib"

FIELDS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome",
          "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]


def load_model(path: str | Path = DEFAULT_MODEL):
    return joblib.load(path)


def predict_one(model, application: dict, threshold: float = 0.5) -> dict:
    """Score a single application. Missing fields are allowed and get imputed."""
    row = pd.DataFrame([{f: application.get(f, np.nan) for f in FIELDS}]).fillna(value=np.nan)
    proba = float(model.predict_proba(row)[0, 1])
    return {"approved": proba >= threshold, "probability": round(proba, 4)}


def predict_batch(model, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out["approval_probability"] = model.predict_proba(df[[c for c in FIELDS if c in df.columns]])[:, 1].round(4)
    out["prediction"] = (out["approval_probability"] >= threshold).map({True: "Approved", False: "Rejected"})
    return out
