import requests
import streamlit as st

st.set_page_config(page_title="Crypto Volatility Predictor", layout="wide")

st.title("Crypto Volatility Predictor")
st.caption("Predicting 14-day volatility of a cryptocurrency based on engineered features.")
image_url = "https://www.shutterstock.com/shutterstock/photos/2591877931/display_1500/stock-photo-best-crypto-exchanges-2591877931.jpg"
st.image(image_url, width=500)

# ─── Cryptocurrency List ───────────────────────────────────────────────────────
CRYPTO_LIST = [
    {"Name": "Bitcoin",       "symbol": "BTC",  "price": 67000.00},
    {"Name": "Ethereum",      "symbol": "ETH",  "price": 3500.00},
    {"Name": "Binance Coin",  "symbol": "BNB",  "price": 580.00},
    {"Name": "Solana",        "symbol": "SOL",  "price": 170.00},
    {"Name": "Ripple",        "symbol": "XRP",  "price": 0.52},
    {"Name": "Cardano",       "symbol": "ADA",  "price": 0.45},
    {"Name": "Dogecoin",      "symbol": "DOGE", "price": 0.16},
    {"Name": "Polkadot",      "symbol": "DOT",  "price": 7.20},
    {"Name": "Litecoin",      "symbol": "LTC",  "price": 85.00},
    {"Name": "Shiba Inu",     "symbol": "SHIB", "price": 0.000024},
    {"Name": "Avalanche",     "symbol": "AVAX", "price": 35.00},
    {"Name": "Chainlink",     "symbol": "LINK", "price": 14.50},
    {"Name": "Polygon",       "symbol": "MATIC","price": 0.85},
    {"Name": "Uniswap",       "symbol": "UNI",  "price": 9.30},
    {"Name": "Tron",          "symbol": "TRX",  "price": 0.12},
]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="https://cryptocurrency-volatility-predictor-2.onrender.com/predict"
)

st.sidebar.markdown("###  Select a Cryptocurrency")

# Dropdown options
crypto_options = [f"{c['Name']} ({c['symbol']}) — ${c['price']}" for c in CRYPTO_LIST]
selected_option = st.sidebar.selectbox("Cryptocurrency:", crypto_options)

# Selected crypto ka data nikalo
selected_index = crypto_options.index(selected_option)
selected_crypto = CRYPTO_LIST[selected_index]

# Selected info show karo
st.sidebar.info(
    f"**Selected:** {selected_crypto['Name']}\n\n"
    f"**Symbol:** {selected_crypto['symbol']}\n\n"
    f"**Current Price:** ${selected_crypto['price']}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Enter The Appropriate Input")

with st.sidebar.form("predict_form"):
    volatility_7        = st.number_input("volatility_7",        value=0.0, format="%.10f")
    ma_7                = st.number_input("ma_7",                value=0.0, format="%.10f")
    ma_14               = st.number_input("ma_14",               value=0.0, format="%.10f")
    hl_range            = st.number_input("hl_range",            value=0.0, format="%.10f")
    volume_mcap_ratio   = st.number_input("volume_mcap_ratio",   value=0.0, format="%.10f")

    submit = st.form_submit_button(" Click To Predict")

# ─── About & Developer ────────────────────────────────────────────────────────
with st.expander("About Us"):
    st.write("""
    **Crypto Volatility Predictor** is a project developed by Mr. Naman Kumar.
    My mission is to provide accurate and actionable insights into the volatility
    of cryptocurrencies, helping traders and investors make informed decisions.
    I leverage advanced machine learning techniques and a comprehensive dataset
    to predict future volatility based on historical trends and engineered features.
    """)

with st.expander("Developer"):
    st.write("""
    **Developer**: Mr. Naman Kumar  
    **GitHub**: https://github.com/Naman27-bit  
    **Email**: namankumar1170@gmail.com
    """)

# ─── Prediction ───────────────────────────────────────────────────────────────
if submit:
    payload = {
        "volatility_7":       float(volatility_7),
        "ma_7":               float(ma_7),
        "ma_14":              float(ma_14),
        "hl_range":           float(hl_range),
        "volume_mcap_ratio":  float(volume_mcap_ratio),
        "dataset_path":       "dataset.csv",
    }

    with st.spinner(f"Predicting volatility for {selected_crypto['Name']} ({selected_crypto['symbol']})..."):
        try:
            r = requests.post(api_url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            pred = data["predicted_volatility_14"]

            st.success(f"Prediction complete for **{selected_crypto['Name']} ({selected_crypto['symbol']})**")
            st.metric("Predicted volatility_14", f"{pred:.10f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("This UI uses engineered features (same 5 inputs as training: volatility_7, ma_7, ma_14, hl_range, volume_mcap_ratio).")
