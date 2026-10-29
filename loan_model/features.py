"""Feature engineering and preprocessing for loan applications."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Loan_Status"
ID_COL = "Loan_ID"

RAW_CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
RAW_NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

ENGINEERED = ["TotalIncome", "LogTotalIncome", "LogLoanAmount", "EMI", "EMI_to_Income", "BalanceIncome"]

ANNUAL_RATE = 0.09


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds affordability features derived from income, loan amount and term.

    - TotalIncome:    applicant + coapplicant monthly income
    - EMI:            monthly instalment at a 9% annual rate (LoanAmount is in thousands)
    - EMI_to_Income:  share of monthly income the EMI takes up
    - BalanceIncome:  income left after paying the EMI
    - Log features:   reduce the heavy right skew of income and loan amount
    """

    def fit(self, X, y=None):
        X = self._coerce(X)
        self.median_loan_ = float(X["LoanAmount"].median())
        self.median_term_ = float(X["Loan_Amount_Term"].median())
        return self

    @staticmethod
    def _coerce(X):
        X = X.copy()
        for col in RAW_NUMERIC:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def transform(self, X):
        X = self._coerce(X)
        loan = X["LoanAmount"].fillna(self.median_loan_)
        term = X["Loan_Amount_Term"].fillna(self.median_term_).clip(lower=12)
        X["TotalIncome"] = X["ApplicantIncome"] + X["CoapplicantIncome"]
        X["LogTotalIncome"] = np.log1p(X["TotalIncome"])
        X["LogLoanAmount"] = np.log1p(loan)
        r = ANNUAL_RATE / 12
        X["EMI"] = loan * 1000 * r * (1 + r) ** term / ((1 + r) ** term - 1)
        # Zero or missing income leaves NaN here; the median imputer downstream fills it.
        X["EMI_to_Income"] = X["EMI"] / X["TotalIncome"].replace(0, np.nan)
        X["BalanceIncome"] = X["TotalIncome"] - X["EMI"]
        return X


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    columns = ColumnTransformer([
        ("num", numeric, RAW_NUMERIC + ENGINEERED),
        ("cat", categorical, RAW_CATEGORICAL),
    ])
    return Pipeline([("engineer", LoanFeatureEngineer()), ("columns", columns)])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    y = (df[TARGET] == "Y").astype(int)
    return X, y
