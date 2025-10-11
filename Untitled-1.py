import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import pandas as pd
import os
from PIL import Image
import base64
import streamlit.components.v1 as components
import numpy as np
import torch
import torch.nn as nn
import joblib

# ---------------------------
# Constants for Transformer
# ---------------------------
SEQUENCE_LENGTH = 30
PREDICT_LENGTH = 7

# ---------------------------
# Setup project base directory
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    layout='wide',
    page_title="Home Page",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Load dataset
# ---------------------------
file_path = os.path.join(BASE_DIR, "Database.xlsx")
df = pd.read_excel(file_path)
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values('Dates')

# ---------------------------
# Load scaler
# ---------------------------
scaler_path = os.path.join(BASE_DIR, "models", "scaler.save")
scaler = joblib.load(scaler_path)

# ---------------------------
# Prepare scaled data
# ---------------------------
feature_columns = ['Consumption', 'Peak Demand', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed', 'solarradiation', 'solarenergy']
data = df[feature_columns].values
data_scaled = scaler.transform(data)

# ---------------------------
# Define Transformer model class
# ---------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x

# ---------------------------
# Load model
# ---------------------------
device = torch.device("cpu")
model_path = os.path.join(BASE_DIR, "models", "transformer_model.pth")
loaded_model = TimeSeriesTransformer(num_features=len(feature_columns))
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.to(device)
loaded_model.eval()

# ---------------------------
# Load Lottie animations
# ---------------------------
with open(os.path.join(BASE_DIR, "Lottie Files", "Animation - 1725291110869.json"), "r") as file:
    url = json.load(file)

with open(os.path.join(BASE_DIR, "Lottie Files", "home.json"), "r") as file:
    url2 = json.load(file)

with open(os.path.join(BASE_DIR, "Lottie Files", "Greenery.json"), "r") as file:
    start_animation = json.load(file)

# ---------------------------
# Background styling
# ---------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Custom navigation buttons
# ---------------------------
st.markdown(
    """
    <style>
    .nav-buttons {
        position: absolute;
        top: 0px;
        right: 0px;
        z-index: 1000;
        display: flex;
        gap: 15px;
    }
    .nav-button {
        background-color: #0E1117;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 8px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        text-transform: uppercase;
        transition: background 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 2px 8px rgba(255, 0, 0, 0.3);
        text-decoration: none;
        letter-spacing: 1px;
    }
    .nav-button:hover {
        color: #fff;
        transform: scale(1.03);
        box-shadow: 0 4px 12px rgba(255, 0, 0, 0.4);
    }
    .nav-button:active {
        transform: scale(1);
        box-shadow: 0 2px 6px rgba(255, 0, 0, 0.3);
    }
    html {
        scroll-behavior: smooth;
    }
    </style>
    """,
    unsafe_allow_html=True
)

scroll_script = """
<script>
    function scrollToSection(id) {
        var element = document.getElementById(id);
        element.scrollIntoView({behavior: 'smooth'});
    }
</script>
"""
components.html(scroll_script)

st.markdown(
    """
    <div class="nav-buttons">
        <a href="#section1" class="nav-button">Predict</a>
        <a href="#section2" class="nav-button">Insights</a>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Font and heading styling
# ---------------------------
st.markdown(
    """
    <style>
        h1 {
            font-size: 2rem;
            margin: 0px;
            padding: 0px;
        }
        .block-container {
            padding-top: 0px;
            margin-top: 0px;
        }
        .centered-text {
            text-align: center;
            font-size: 2.5rem;
            margin: 0;
            padding: 0px;
        }
        .custom-container {
            padding: 0px;
            margin: 0 auto;
        }
        .lottie-container {
            margin: 0px auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Main headings
# ---------------------------
st.markdown("<h1 style='text-align: center;'>Unveiling Delhi’s Electricity Needs</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>One <span style='color: red;'>Prediction</span> at a Time</h1>", unsafe_allow_html=True)

# ---------------------------
# Video embedding
# ---------------------------
video_path = os.path.join(BASE_DIR, "Video", "New Delhi Map Cut.mp4")

if os.path.exists(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode()

    video_html = f"""
        <video autoplay muted loop width="800" playsinline>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    """

    st.markdown(video_html, unsafe_allow_html=True)
else:
    st.error(f"Video file not found at {video_path}. Please check the file path.")

# ---------------------------
# Section 1: Prediction
# ---------------------------
st.markdown(
    """
    <div id="section1">
    <style>
        .subheader {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            position: relative;
            padding: 60px 0px 0px 0px;
            display: inline-block;
            color: #ffffff;
            animation: float 2s ease-in-out infinite, glow 1.5s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0); }
        }
        @keyframes glow {
            0% { text-shadow: 0 0 10px #00bfff; }
            50% { text-shadow: 0 0 20px #00bfff; }
            100% { text-shadow: 0 0 10px #00bfff; }
        }
    </style>
    <h2 class="subheader">Predicting the Pulse of Delhi’s Power Needs</h2>
    </div>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.write("-----")
    left_column, right_column = st.columns(2)
    with left_column:
        date = st.date_input("ENTER DATE: ")
    with right_column:
        options = ["OVERALL", "CENTRAL", "NORTH", "SOUTH", "EAST", "WEST",
                   "NORTH-EAST", "NORTH-WEST", "SOUTH-EAST", "SOUTH-WEST"]
        locality = st.selectbox("ENTER LOCALITY: ", options)

# ---------------------------
# Prediction display function
# ---------------------------
def display_labels_and_values(date, locality, predictions=None, feature_values=None):
    st.markdown(
        """
        <style>
            .hover-container { transition: all 0.3s ease; font-size: 1.2rem; margin-bottom: 10px; }
            .label { color: white; font-weight: bold; transition: color 0.3s ease; }
            .hover-container:hover .label { color: #00BFFF; }
            .value { color: #ffffff; font-weight: bold; transition: color 0.3s ease; }
            .hover-container:hover .value { color: #FFDB58; }
            .hover-container:hover { font-size: 1.3rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if predictions is not None and feature_values is not None:
        st.markdown(f"<div class='hover-container'><span class='label'>⚡Estimated Consumption: </span><span class='value'>{predictions[0] * prop:.2f} MUs</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hover-container'><span class='label'>🌡️Estimated Average Temperature: </span><span class='value'>{feature_values.iloc[0]['temp']}\u00B0C</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hover-container'><span class='label'>🔥Estimated Maximum Temperature: </span><span class='value'>{feature_values.iloc[0]['tempmax']}\u00B0C</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hover-container'><span class='label'>❄️Estimated Minimum Temperature: </span><span class='value'>{feature_values.iloc[0]['tempmin']}\u00B0C</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hover-container'><span class='label'>💧Estimated Humidity: </span><span class='value'>{feature_values.iloc[0]['humidity']} %</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hover-container'><span class='label'>🍃Estimated Wind Speed: </span><span class='value'>{feature_values.iloc[0]['windspeed']} Km/Hr</span></div>", unsafe_allow_html=True)

display_labels_and_values(None, None)

if st.button("Submit"):
    prop = 1
    target_date = pd.to_datetime(date)
    latest_date_in_dataset = df['Dates'].max()
    oldest_date_in_dataset = df['Dates'].min()

    adjusted_date = target_date
    if target_date > latest_date_in_dataset:
        latest_year = latest_date_in_dataset.year
        adjusted_date = target_date.replace(year=latest_year)
    elif target_date < oldest_date_in_dataset:
        oldest_year = oldest_date_in_dataset.year
        adjusted_date = target_date.replace(year=oldest_year)

    feature_values = df.loc[
        (df['Dates'] == adjusted_date),
        ['Dates', 'temp', 'tempmax', 'tempmin', 'humidity', 'windspeed']
    ]

    if feature_values.empty:
        st.write(f"No data found for {adjusted_date.date()}")
    else:
        target_idx = df[df['Dates'] == adjusted_date].index[0]
        seq_start = target_idx - SEQUENCE_LENGTH
        if seq_start < 0:
            st.write("Not enough historical data for this date.")
        else:
            locality_prop_map = {
                "OVERALL": 1, "CENTRAL": 0.09, "NORTH": 0.11, "SOUTH": 0.18,
                "EAST": 0.10, "WEST": 0.16, "NORTH-EAST": 0.06, "NORTH-WEST": 0.15,
                "SOUTH-EAST": 0.05, "SOUTH-WEST": 0.10
            }
            prop = locality_prop_map.get(locality, 1)
            seq = data_scaled[seq_start:target_idx]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = loaded_model(seq_tensor)
                pred_cons_scaled = output[:, SEQUENCE_LENGTH - PREDICT_LENGTH, 0].item()
            dummy_row = [pred_cons_scaled] + [0] * 8
            pred_cons = scaler.inverse_transform([dummy_row])[0][0]
            predictions = [pred_cons]
            display_labels_and_values(date, locality, predictions, feature_values)

# ---------------------------
# Section 2: Insights
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.image(os.path.join(BASE_DIR, "Images", "Map.jpg"), caption="Natural Load Growth Map", use_container_width=True)

st.markdown(
    """
    <div id="section2">
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            padding: 10px 0px 0px 0px;
            text-align: left;
            transition: all 0.6s ease;
            transform: perspective(500px);
        }
        .title:hover {
            animation: rotate-scale 1s forwards;
        }
        @keyframes rotate-scale {
            0% { transform: perspective(500px) scale(1) rotateX(0deg) rotateY(0deg); }
            50% { transform: perspective(500px) scale(1.05) rotateX(5deg) rotateY(5deg); }
            100% { transform: perspective(500px) scale(1.1) rotateX(-5deg) rotateY(-5deg); }
        }
    </style>
    <h1 class="title"><span style='color: #66ff00;'>Save 3,000 GWh, </span><span style='color: red;'>Cut ₹1,000 Crore:</span> Delhi’s Energy Revolution!</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .custom-font { font-size: 25px; transition: all 0.3s ease; }
        .custom-font:hover { font-size: 25px; color: #FF8C00; }
    </style>
    <ul style='list-style-type: disc; padding-left: 20px;'>
        <li><h4 class='custom-font'><span style='color: #87CEEB;'>Solar Power Push 🌞</span><br>Delhi can meet 20% of its electricity demand through rooftop solar, saving ₹1,000 crore annually and reducing 1,500 MW of non-renewable load.</h4></li>
        <li><h4 class='custom-font'><span style='color: #87CEEB;'>Industrial Energy Efficiency ⚙️</span><br>Upgrading Delhi’s industrial plants with energy-efficient machinery and shifting to off-peak hours could save ₹2-3 crore annually per plant.</h4></li>
        <li><h4 class='custom-font'><span style='color: #87CEEB;'>Smart Metering & Consumption Shifts 📊</span><br>Deploy smart meters across 5.3 million homes to reduce wastage and shift consumption to off-peak hours.</h4></li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='text-align: center;'>💡Powering Tomorrow with Less Today🌱</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888888;
            padding: 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='footer'>© 2024 Delhi Energy Prediction, All Rights Reserved</div>", unsafe_allow_html=True)






# # Calculate metrics using one-step ahead predictions
# y_true = data[SEQUENCE_LENGTH:, 0]  # Consumption from SEQUENCE_LENGTH onwards
# y_pred = []

# for i in range(SEQUENCE_LENGTH, len(data_scaled)):
#     seq = data_scaled[i - SEQUENCE_LENGTH:i]
#     seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = loaded_model(seq_tensor)
#         pred_cons_scaled = output[:, SEQUENCE_LENGTH - PREDICT_LENGTH, 0].item()
#     dummy_row = [pred_cons_scaled] + [0] * 8
#     pred_cons = scaler.inverse_transform([dummy_row])[0][0]
#     y_pred.append(pred_cons)

# y_pred = np.array(y_pred)

# # Calculate metrics
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else 0
# r2 = r2_score(y_true, y_pred)

# # Display metrics in Streamlit
# st.subheader("📊 Transformer Model Performance Metrics")
# st.write(f"**MAE:** {mae:.2f}")
# st.write(f"**RMSE:** {rmse:.2f}")
# st.write(f"**MAPE:** {mape:.2f}%")
# st.write(f"**R² Score:** {r2:.4f}")

# # Optional dataframe display
# metrics_df = pd.DataFrame({
#     "Metric": ["MAE", "RMSE", "MAPE", "R²"],
#     "Value": [mae, rmse, mape, r2]
# })
# st.dataframe(metrics_df)