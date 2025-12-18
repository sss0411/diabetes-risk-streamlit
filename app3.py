import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load model artifacts
# ==============================
model_bal = joblib.load("diabetes_model_balanced.joblib")
scaler = joblib.load("scaler.joblib")
features = joblib.load("features.joblib")

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Diabetes Risk Screening Tool",
    layout="centered"
)

# ==============================
# Header
# ==============================
st.markdown("## 🩺 Diabetes Risk Screening Tool")
st.write(
    "This tool estimates the risk of diabetes based on common health and lifestyle factors."
)
st.warning(
    "The result is for screening purposes only and does not provide a medical diagnosis."
)

st.markdown("---")

# ==============================
# Patient information
# ==============================
st.markdown("### Patient information")

age = st.number_input("Age (years)", min_value=0, max_value=120, value=23)

sex = st.selectbox("Sex", ["Female", "Male"])
education = st.selectbox("Education level", ["Low", "Medium", "High"])
marital = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Widowed"])
labor = st.selectbox("Labor status", ["Employed", "Unemployed"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol drinking", ["No", "Yes"])
physical_inactivity = st.selectbox("Physical inactivity", ["No", "Yes"])
salt = st.selectbox("High salt intake", ["No", "Yes"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=19.0)
waist = st.number_input("Waist circumference (cm)", min_value=40, max_value=200, value=60)

# ==============================
# Encode inputs (NO Obesity)
# ==============================
input_dict = {
    "Age": age,
    "Sex": 1 if sex == "Male" else 0,
    "Education_level": {"Low": 0, "Medium": 1, "High": 2}[education],
    "Marital_status": {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}[marital],
    "Labor_status": 1 if labor == "Employed" else 0,
    "Smoking": 1 if smoking == "Yes" else 0,
    "Alcohol_drinking": 1 if alcohol == "Yes" else 0,
    "Physical_inactivity": 1 if physical_inactivity == "Yes" else 0,
    "High_salt_intake": 1 if salt == "Yes" else 0,
    "BMI": bmi,
    "Waist_circumference": waist
}

# ==============================
# Prepare model input (SAFE)
# ==============================
X = pd.DataFrame(columns=features)
X.loc[0] = [input_dict[f] for f in features]

# ==============================
# Prediction
# ==============================
if st.button("Predict risk"):
    X_scaled = scaler.transform(X)
    prob = model_bal.predict_proba(X_scaled)[0][1]

    st.markdown("### Result")
    st.write(f"Estimated probability of diabetes: **{prob:.2f}**")

    if prob < 0.3:
        st.success("✅ Low risk of diabetes (screen-negative)")
    elif prob < 0.6:
        st.warning("⚠️ Moderate risk of diabetes")
    else:
        st.error("❗ High risk of diabetes (screen-positive)")

# ==============================
# Footer
# ==============================
st.markdown(
    "<hr><div style='text-align:center; color:gray; font-size:12px;'>"
    "Created by SanimgulSS</div>",
    unsafe_allow_html=True
)
