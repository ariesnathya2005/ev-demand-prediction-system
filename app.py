import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="EV Demand Prediction", layout="centered")

st.title("⚡ EV Charging Demand Predictor")

# Load compiled model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Form Inputs
st.header("Make a Prediction")
col1, col2, col3, col4 = st.columns(4)

with col1:
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
with col2:
    day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=15)
with col3:
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
with col4:
    dayofweek = st.number_input("Day of Week (0-6)", min_value=0, max_value=6, value=2, help="0=Monday, 6=Sunday")

# Prediction Button
if st.button("Predict"):
    features = np.array([[hour, day, month, dayofweek]])
    prediction = model.predict(features)
    st.success(f"### Predicted Charging Demand: {round(prediction[0], 2)}")

st.markdown("---")

# Data Visualization
st.header("📊 Data Visualization (Demand vs Time)")

@st.cache_data
def get_chart_data():
    df = pd.read_csv("ev_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    return df.groupby('hour')['demand'].mean()

hourly_demand = get_chart_data()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hourly_demand.index, hourly_demand.values, color='b', marker='o')
ax.set_title("Average EV Charging Demand vs Time (Hour of Day)")
ax.set_xlabel("Hour of Day (0-23)")
ax.set_ylabel("Average Charging Demand")
ax.grid(True)

# Render plot in Streamlit
st.pyplot(fig)
