"""Generate a synthetic loan application dataset.

The schema matches the widely used "Loan Prediction" dataset (Analytics Vidhya / Kaggle),
so you can drop the real train.csv into data/loans.csv and everything still works.

    python scripts/generate_dataset.py --rows 2000

Approval is driven mainly by credit history and the affordability of the EMI relative
to household income, with noise and some missing values, like real application data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "loans.csv"


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n, p=[0.8, 0.2])
    married = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.57, 0.17, 0.17, 0.09])
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], n, p=[0.86, 0.14])
    area = rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.33, 0.38, 0.29])

    # Monthly incomes, right-skewed like real salaries.
    applicant_income = np.round(rng.lognormal(8.3, 0.55, n) * np.where(education == "Graduate", 1.15, 0.85))
    has_coapplicant = (married == "Yes") & (rng.random(n) < 0.7)
    coapplicant_income = np.where(has_coapplicant, np.round(rng.lognormal(7.6, 0.6, n)), 0)

    total_income = applicant_income + coapplicant_income
    # Loan amount in thousands, loosely tied to income.
    loan_amount = np.clip(np.round(total_income / 1000 * rng.uniform(15, 40, n) + rng.normal(0, 25, n)), 9, 700)
    term = rng.choice([360, 180, 480, 300, 240, 120, 84, 60], n, p=[0.83, 0.07, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
    credit_history = (rng.random(n) < 0.84).astype(float)

    # EMI and the share of income it takes up drive approval.
    r = 0.09 / 12
    emi = loan_amount * 1000 * r * (1 + r) ** term / ((1 + r) ** term - 1)
    emi_ratio = emi / total_income

    logit = (
        -1.6
        + 3.0 * credit_history
        - 9.0 * np.clip(emi_ratio - 0.2, 0, None)
        + 0.35 * (education == "Graduate")
        + 0.45 * (area == "Semiurban") - 0.25 * (area == "Rural")
        + 0.25 * (married == "Yes")
        - 0.15 * (dependents == "3+")
        + rng.normal(0, 0.7, n)
    )
    approved = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "Loan_ID": [f"LP{100000 + i}" for i in range(n)],
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income.astype(int),
        "CoapplicantIncome": coapplicant_income.astype(int),
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": term.astype(float),
        "Credit_History": credit_history,
        "Property_Area": area,
        "Loan_Status": np.where(approved, "Y", "N"),
    })

    # Missing values in the same columns the real dataset has gaps in.
    for col, rate in [("Gender", 0.02), ("Married", 0.005), ("Dependents", 0.025), ("Self_Employed", 0.05),
                      ("LoanAmount", 0.035), ("Loan_Amount_Term", 0.02), ("Credit_History", 0.08)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    df = generate(args.rows, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out} (approval rate {df['Loan_Status'].eq('Y').mean():.1%})")


if __name__ == "__main__":
    main()
