import streamlit as st
import pandas as pd
import joblib
import json
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = "shahdadpuri/predictive-maintenance-engine-model"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.joblib", repo_type="model")
    model = joblib.load(model_path)
    info = {}
    try:
        info_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model_info.json", repo_type="model")
        with open(info_path) as f:
            info = json.load(f)
    except Exception:
        pass
    return model, info


model, info = load_model()

st.title("Engine Predictive Maintenance")
st.write(
    "Enter the current sensor readings below to check whether the engine is "
    "likely **Normal** or **Faulty**."
)

if info:
    st.caption(
        f"Model in use: **{info.get('algorithm', 'N/A')}**  |  "
        f"Test F1-score: **{info.get('F1', 'N/A')}**"
    )

col1, col2 = st.columns(2)
with col1:
    rpm = st.number_input("Engine RPM", min_value=0.0, max_value=3000.0, value=800.0, step=1.0)
    oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.3, step=0.01)
    fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.5, step=0.01)
with col2:
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.3, step=0.01)
    oil_temp = st.number_input("Lub Oil Temperature (deg C)", min_value=0.0, max_value=120.0, value=78.0, step=0.1)
    coolant_temp = st.number_input("Coolant Temperature (deg C)", min_value=0.0, max_value=130.0, value=79.0, step=0.1)

input_df = pd.DataFrame([{
    "Engine rpm": rpm,
    "Lub oil pressure": oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": oil_temp,
    "Coolant temp": coolant_temp,
}])

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_df)[0]
    proba_faulty = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"Prediction: FAULTY (model confidence: {proba_faulty:.0%})")
    else:
        st.success(f"Prediction: NORMAL (model confidence engine is faulty: {proba_faulty:.0%})")

st.caption("Model loaded live from the Hugging Face model hub: " + HF_MODEL_REPO)
