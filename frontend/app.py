from pathlib import Path
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "https://fraud-api-gef5.onrender.com"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
)

st.title("💳 Fraud Detection System")


file_path = SAMPLE_DATA_DIR / "sample_data.csv"

if file_path.suffix == ".csv":
    sample_df = pd.read_csv(file_path)
elif file_path.suffix == ".json":
    sample_df = pd.read_json(file_path)
else:
    st.error("Unsupported sample data file format.")
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📥 Download CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )

with col2:
    st.download_button(
        label="📥 Download JSON",
        data=sample_df.to_json(orient="records"),
        file_name="sample_transactions.json",
        mime="application/json",
    )

st.info("Copy the transactions from CSV/JSON into a CSV file with the same format as the sample file.")


st.sidebar.markdown("### API Status")

try:
    health = requests.get(f"{API_URL}/health", timeout=10)
    health.raise_for_status()
    health_data = health.json()

    st.sidebar.success("API Connected ✅")
    st.sidebar.write("Loaded Models:", health_data.get("loaded_models", []))
    st.sidebar.write("Load Status:", health_data.get("load_status", {}))
except requests.RequestException:
    st.sidebar.error("API Not Connected ❌")


model_choice = st.sidebar.selectbox(
    "Select Model",
    ["catboost", "balanced_rf", "hybrid_catboost", "hybrid_lr"],
)


st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step", value=740)
    amount = st.number_input("Amount", value=99999999.99)
    oldbalanceOrg = st.number_input("Old Balance Origin", value=99999999.99)
    newbalanceOrig = st.number_input("New Balance Origin", value=99999999.99)

with col2:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT","TRANSFER", "PAYMENT", "DEBIT"],
    )
    oldbalanceDest = st.number_input("Old Balance Destination", value=99999999.99)
    newbalanceDest = st.number_input("New Balance Destination", value=99999999.99)

nameOrig = st.text_input("Sender ID", "C123456789")
nameDest = st.text_input("Receiver ID", "M9876543210", max_chars=11)


if st.button("🔍 Predict Fraud"):
    input_data = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": nameOrig,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict/{model_choice}",
            json=input_data,
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()

        st.subheader("📊 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", result["decision"])

        with col2:
            fraud_risk_score = result.get("fraud_risk_score")
            display_score = f"{fraud_risk_score:.4f}" if fraud_risk_score is not None else "N/A"
            st.metric("Fraud Risk Score", display_score)

        with col3:
            st.metric("Threshold Used", result["threshold_used"])

        if "top_reasons" in result and result["top_reasons"]:
            st.subheader("🧠 Top Fraud Reasons")
            for reason in result["top_reasons"]:
                st.write(f"🔹 {reason['feature']} → Impact: {reason['impact']:.4f}")

        with st.expander("🔎 Full Response"):
            st.json(result)

    except requests.RequestException as e:
        st.error(f"Error connecting to API: {e}")
