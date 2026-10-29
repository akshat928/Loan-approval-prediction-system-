"""Streamlit app for real-time loan approval predictions.

    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from loan_model.predict import FIELDS, load_model, predict_batch, predict_one

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_model.joblib"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m loan_model.train` first.")
    st.stop()

model = get_model()

st.title("🏦 Loan Approval Predictor")
st.caption("Enter applicant details to get an instant approval estimate.")

tab_single, tab_batch, tab_insights = st.tabs(["Single application", "Batch (CSV)", "Model insights"])

with tab_single:
    with st.form("application"):
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        married = c2.selectbox("Married", ["Yes", "No"])
        dependents = c1.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = c2.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = c1.selectbox("Self employed", ["No", "Yes"])
        area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

        st.markdown("**Financials**")
        c3, c4 = st.columns(2)
        applicant_income = c3.number_input("Applicant monthly income (₹)", min_value=0, value=5000, step=500)
        coapplicant_income = c4.number_input("Co-applicant monthly income (₹)", min_value=0, value=0, step=500)
        loan_amount = c3.number_input("Loan amount (₹ thousands)", min_value=1, value=120, step=5)
        term = c4.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 84, 60, 480])
        credit_history = st.radio("Meets credit history guidelines?", ["Yes", "No"], horizontal=True)

        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        application = {
            "Gender": gender, "Married": married, "Dependents": dependents, "Education": education,
            "Self_Employed": self_employed, "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income, "LoanAmount": loan_amount,
            "Loan_Amount_Term": float(term), "Credit_History": 1.0 if credit_history == "Yes" else 0.0,
            "Property_Area": area,
        }
        result = predict_one(model, application)
        total = applicant_income + coapplicant_income
        r = 0.09 / 12
        emi = loan_amount * 1000 * r * (1 + r) ** term / ((1 + r) ** term - 1)

        if result["approved"]:
            st.success(f"✅ Likely **approved**: {result['probability']:.0%} approval probability")
        else:
            st.error(f"❌ Likely **rejected**: {result['probability']:.0%} approval probability")
        st.progress(result["probability"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Total income", f"₹{total:,.0f}")
        m2.metric("Estimated EMI", f"₹{emi:,.0f}")
        m3.metric("EMI / income", f"{emi / total:.0%}" if total else "n/a")
        if total and emi / total > 0.5:
            st.warning("The EMI takes up more than half of monthly income, which strongly lowers approval odds.")
        if credit_history == "No":
            st.warning("Not meeting credit history guidelines is the single biggest driver of rejection.")

with tab_batch:
    st.write(f"Upload a CSV with these columns: `{', '.join(FIELDS)}`")
    upload = st.file_uploader("Applications CSV", type="csv")
    if upload:
        df = pd.read_csv(upload)
        missing = [c for c in FIELDS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            scored = predict_batch(model, df)
            st.dataframe(scored, width="stretch")
            st.download_button("Download predictions", scored.to_csv(index=False), "predictions.csv", "text/csv")

with tab_insights:
    metrics_path = ROOT / "models" / "metrics.json"
    imp_path = ROOT / "models" / "feature_importance.csv"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.subheader(f"Model: {m['best_model'].replace('_', ' ').title()}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{m['test']['accuracy']:.1%}")
        c2.metric("Test F1", f"{m['test']['f1']:.3f}")
        c3.metric("Test ROC-AUC", f"{m['test']['roc_auc']:.3f}")
        st.caption("Cross-validated ROC-AUC by model")
        st.dataframe(pd.DataFrame(m["cv"]).T, width="stretch")
    if imp_path.exists():
        st.subheader("What drives approval")
        st.caption("Drop in ROC-AUC when each input is randomly shuffled (permutation importance).")
        imp = pd.read_csv(imp_path).set_index("feature")["importance"].clip(lower=0)
        st.bar_chart(imp)
