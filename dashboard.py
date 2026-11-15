
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Sentinel | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide"
)

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

if not API_URL.startswith(("http://", "https://")):
    API_URL = f"http://{API_URL}"
# ---------------- SESSION STATE ----------------

if "transactions" not in st.session_state:
    st.session_state.transactions = []

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #101c30);
    color: #f1f5f9;
}
[data-testid="stMetric"] {
    background: #17263b;
    padding: 20px;
    border: 1px solid #293b55;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    color: #38bdf8;
}
section[data-testid="stSidebar"] {
    background: #0b1728;
}
.stButton > button {
    background: #087ea4;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background: #06a6cf;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🛡️ SENTINEL")
    st.caption("AI FRAUD INTELLIGENCE PLATFORM")
    st.divider()

    st.subheader("System Status")

    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.ok:
            st.success("Prediction API Online")
        else:
            st.warning("API returned an unexpected status")
    except requests.RequestException:
        st.error("Prediction API Offline")

    st.divider()
    st.caption("Machine Learning Engine")
    st.write("Random Forest + SMOTE")
    st.caption("Environment")
    st.write("Development / Demo")

    if st.button("Clear Session History"):
        st.session_state.transactions = []
        st.rerun()

# ---------------- HEADER ----------------

st.title("🛡️ Fraud Intelligence Center")
st.markdown(
    "Real-time transaction monitoring powered by machine learning."
)

st.divider()

# ---------------- KPI METRICS ----------------

history = st.session_state.transactions

total = len(history)
fraud_count = sum(
    item["prediction"] == 1 for item in history
)
legitimate_count = total - fraud_count

average_risk = (
    sum(item["probability"] for item in history) / total
    if total else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions Analyzed", total)
c2.metric("Potential Fraud", fraud_count)
c3.metric("Likely Legitimate", legitimate_count)
c4.metric("Average Fraud Probability", f"{average_risk:.1%}")

st.divider()

# ---------------- TRANSACTION ANALYSIS ----------------

left, right = st.columns([1, 1])

with left:
    st.subheader("🔍 Transaction Risk Analysis")
    st.caption("Enter transaction details for AI-powered assessment.")

    with st.form("fraud_analysis_form"):

        amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=1.0,
            value=850.0,
            step=100.0
        )

        hour = st.slider(
            "Transaction Hour",
            0, 23, 2
        )

        distance = st.number_input(
            "Distance From Home (KM)",
            min_value=0.0,
            value=120.0
        )

        count = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            value=8
        )

        risk = st.slider(
            "Merchant Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.9
        )

        submitted = st.form_submit_button(
            "⚡ Analyze Transaction",
            use_container_width=True
        )

    if submitted:

        transaction = {
            "amount": amount,
            "hour": hour,
            "distance_from_home_km": distance,
            "transactions_last_24h": count,
            "merchant_risk_score": risk
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            probability = float(result["fraud_probability"])
            prediction = int(result["prediction"])

            record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
                "hour": hour,
                "distance_km": distance,
                "transactions_24h": count,
                "merchant_risk": risk,
                "probability": probability,
                "prediction": prediction,
                "label": result.get("label", "unknown")
            }

            st.session_state.transactions.append(record)

            st.session_state.latest_result = record
            st.rerun()

        except requests.RequestException as error:
            st.error(f"API connection failed: {error}")
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Unexpected prediction response: {error}")

with right:
    st.subheader("📊 Latest Risk Assessment")

    latest = st.session_state.get("latest_result")

    if latest:

        st.metric(
            "Fraud Probability",
            f"{latest['probability']:.2%}"
        )

        st.progress(
            max(0.0, min(1.0, latest["probability"]))
        )

        if latest["prediction"] == 1:
            st.error("⚠️ Potential Fraud Detected")
            st.write(
                "The model classified this transaction as potential fraud."
            )
        else:
            st.success("✅ Likely Legitimate")
            st.write(
                "The model classified this transaction as likely legitimate."
            )

        st.caption(f"Model label: {latest['label']}")

    else:
        st.info(
            "Submit a transaction to see its fraud probability and risk assessment."
        )

# ---------------- ANALYTICS ----------------

st.divider()
st.subheader("📈 Transaction Analytics")

if history:

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Transaction Classification**")

        classification = pd.DataFrame({
            "Category": ["Potential Fraud", "Likely Legitimate"],
            "Count": [fraud_count, legitimate_count]
        }).set_index("Category")

        st.bar_chart(classification)

    with chart_col2:
        st.markdown("**Fraud Probability by Transaction**")

        probabilities = pd.DataFrame({
            "Transaction": [
                f"TXN {i + 1}" for i in range(total)
            ],
            "Fraud Probability": [
                item["probability"] for item in history
            ]
        }).set_index("Transaction")

        st.line_chart(probabilities)

else:
    st.info("Analytics will appear after you analyze your first transaction.")

# ---------------- TRANSACTION HISTORY ----------------

st.divider()
st.subheader("📋 Transaction History")

if history:

    table = pd.DataFrame(history)

    table["classification"] = table["prediction"].map({
        1: "Potential Fraud",
        0: "Likely Legitimate"
    })

    display_table = table[[
        "time",
        "amount",
        "hour",
        "probability",
        "classification"
    ]].copy()

    display_table["probability"] = (
        display_table["probability"] * 100
    ).round(2).astype(str) + "%"

    display_table = display_table.rename(columns={
        "time": "Date & Time",
        "amount": "Amount (₹)",
        "hour": "Hour",
        "probability": "Fraud Probability",
        "classification": "Classification"
    })

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    csv = table.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Transaction Report",
        data=csv,
        file_name="fraud_transaction_report.csv",
        mime="text/csv"
    )

else:
    st.info("No transactions recorded in this session yet.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "SENTINEL AI | Educational fraud detection demonstration. "
    "Model trained on synthetic data. Not for real financial decisions."
)
