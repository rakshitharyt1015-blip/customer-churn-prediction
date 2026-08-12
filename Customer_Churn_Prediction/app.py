import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to estimate whether the customer may leave the service.")

model_path = Path("churn_model.joblib")
if not model_path.exists():
    st.warning("Model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(model_path)

age = st.number_input("Age", 18, 100, 30)
contract = st.selectbox("Contract", ["Monthly", "Yearly"])
plan = st.selectbox("Plan", ["Basic", "Premium"])
monthly = st.number_input("Monthly Charges", 10.0, 200.0, 50.0)
tenure = st.number_input("Tenure (months)", 0, 120, 12)

if st.button("Predict Churn"):
    row = pd.DataFrame([{
        "Age": age,
        "Contract": contract,
        "Plan": plan,
        "MonthlyCharges": monthly,
        "TenureMonths": tenure
    }])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.error(f"⚠️ Likely to churn — probability: {prob:.1%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {prob:.1%}")
