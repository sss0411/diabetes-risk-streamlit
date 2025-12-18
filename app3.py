import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("diabetes_model.joblib")
model_bal = joblib.load("diabetes_model_balanced.joblib")
scaler = joblib.load("scaler.joblib")
features = joblib.load("features.joblib")

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("This tool estimates diabetes risk based on clinical risk factors.")

# Select model mode
mode = st.radio(
    "Select screening mode:",
    ["Moderate sensitivity (clinical)", "High sensitivity (population)"]
)

# Input form
user_input = {}
for feature in features:
    if feature.lower() in ["sex", "hypertension", "physical_inactivity"]:
        user_input[feature] = st.selectbox(feature, [0, 1])
    else:
        user_input[feature] = st.number_input(feature, min_value=0.0)

# Predict
if st.button("Predict risk"):
    X = pd.DataFrame([user_input])
    X_scaled = scaler.transform(X)

    if mode.startswith("Moderate"):
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        prob = model_bal.predict_proba(X_scaled)[0][1]

    st.subheader("Prediction result")
    st.write(f"Estimated diabetes risk probability: **{prob:.2f}**")

    if prob >= 0.5:
        st.error("High risk of diabetes")
    else:
        st.success("Low risk of diabetes")
st.markdown(
    "<hr><center>Created by SanimgulSS</center>",
    unsafe_allow_html=True
)
