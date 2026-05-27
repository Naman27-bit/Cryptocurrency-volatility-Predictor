import requests
import streamlit as st

st.set_page_config(page_title="Crypto Volatility Predictor", layout="wide")

st.title("Crypto Volatility Predictor")
st.caption("Predicting 14-day volatility of a cryptocurrency based on engineered features.")

api_url = st.sidebar.text_input("FastAPI URL", value="https://cryptocurrency-volatility-predictor-2.onrender.com/predict")

st.sidebar.markdown("### Enter The Appropriate Input")

with st.sidebar.form("predict_form"):
    volatility_7 = st.number_input("volatility_7", value=0.0, format="%.10f")
    ma_7 = st.number_input("ma_7", value=0.0, format="%.10f")
    ma_14 = st.number_input("ma_14", value=0.0, format="%.10f")
    hl_range = st.number_input("hl_range", value=0.0, format="%.10f")
    volume_mcap_ratio = st.number_input("volume_mcap_ratio", value=0.0, format="%.10f")

    submit = st.form_submit_button(" Click To Predict")

if submit:
    payload = {
        "volatility_7": float(volatility_7),
        "ma_7": float(ma_7),
        "ma_14": float(ma_14),
        "hl_range": float(hl_range),
        "volume_mcap_ratio": float(volume_mcap_ratio),
        "dataset_path": "dataset.csv",
    }
    with st.spinner("Predicting..."):
        try:
            r = requests.post(api_url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            pred = data["predicted_volatility_14"]
            st.metric("Predicted volatility_14", f"{pred:.10f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("This UI uses engineered features (same 5 inputs as training: volatility_7, ma_7, ma_14, hl_range, volume_mcap_ratio).")