import json

import numpy as np
import pandas as pd
import pytest

from loan_model.features import LoanFeatureEngineer, build_preprocessor, split_xy
from loan_model.predict import load_model, predict_batch, predict_one
from loan_model.train import train
from scripts.generate_dataset import generate

GOOD = {"Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "ApplicantIncome": 9000, "CoapplicantIncome": 3000,
        "LoanAmount": 100, "Loan_Amount_Term": 360.0, "Credit_History": 1.0, "Property_Area": "Semiurban"}


def test_generate_dataset_schema():
    df = generate(300, seed=1)
    assert len(df) == 300
    assert set(df["Loan_Status"].unique()) <= {"Y", "N"}
    assert df["Credit_History"].isna().any()  # has realistic gaps


def test_feature_engineer_emi():
    df = pd.DataFrame([{"ApplicantIncome": 5000, "CoapplicantIncome": 0,
                        "LoanAmount": 100, "Loan_Amount_Term": 360.0}])
    out = LoanFeatureEngineer().fit(df).transform(df)
    # ₹1,00,000 at 9% for 30 years is about ₹805/month.
    assert out["EMI"].iloc[0] == pytest.approx(804.6, abs=1)
    assert out["EMI_to_Income"].iloc[0] == pytest.approx(0.161, abs=0.01)
    assert out["TotalIncome"].iloc[0] == 5000


def test_preprocessor_handles_missing_values():
    df = generate(200, seed=2)
    X, _ = split_xy(df)
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt)).any()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    data = d / "loans.csv"
    generate(1200, seed=7).to_csv(data, index=False)
    report = train(str(data), str(d / "models"))
    return report, d / "models"


def test_training_beats_baseline(trained):
    report, _ = trained
    assert report["test"]["roc_auc"] > 0.7
    assert report["best_model"] in report["cv"]


def test_artifacts_saved(trained):
    _, models = trained
    assert (models / "loan_model.joblib").exists()
    metrics = json.loads((models / "metrics.json").read_text())
    assert "confusion_matrix" in metrics["test"]
    imp = pd.read_csv(models / "feature_importance.csv")
    # Credit history should be the strongest driver, as it is in the data.
    assert imp.iloc[0]["feature"] == "Credit_History"


def test_predictions_follow_credit_history(trained):
    _, models = trained
    model = load_model(models / "loan_model.joblib")
    good = predict_one(model, GOOD)
    bad = predict_one(model, {**GOOD, "Credit_History": 0.0})
    assert good["probability"] > bad["probability"]
    assert good["approved"]


def test_predict_one_with_missing_fields(trained):
    _, models = trained
    model = load_model(models / "loan_model.joblib")
    partial = {k: v for k, v in GOOD.items() if k not in ("Gender", "LoanAmount")}
    result = predict_one(model, partial)
    assert 0 <= result["probability"] <= 1


def test_predict_batch(trained):
    _, models = trained
    model = load_model(models / "loan_model.joblib")
    out = predict_batch(model, pd.DataFrame([GOOD, {**GOOD, "Credit_History": 0.0}]))
    assert list(out["prediction"].unique()) != []
    assert "approval_probability" in out.columns
