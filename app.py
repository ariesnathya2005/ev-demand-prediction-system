import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# CONFIG & CACHING
# -------------------------------
st.set_page_config(page_title="EV Demand Prediction", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("ev_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

model = load_model()
df = load_data()

# -------------------------------
# UI HEADER & PROJECT DESC
# -------------------------------
st.write("""
## ⚡ EV Charging Demand Prediction System

This project predicts electric vehicle charging demand using machine learning.

### Features:
- Real-time demand prediction
- Time-based forecasting
- Data visualization
- Cloud deployment using Render
""")

st.markdown("---")

# -------------------------------
# USER INPUT
# -------------------------------
st.write("### 🎛️ Input Parameters")
col1, col2, col3, col4 = st.columns(4)
with col1:
    hour = st.slider("Hour", 0, 23, 12)
with col2:
    day = st.slider("Day", 1, 31, 15)
with col3:
    month = st.slider("Month", 1, 12, 6)
with col4:
    dayofweek = st.slider("Day of Week", 0, 6, 3, help="0=Mon, 6=Sun")

# -------------------------------
# PREDICTION & WARNINGS
# -------------------------------
if st.button("Predict Demand"):
    features = np.array([[hour, day, month, dayofweek]])
    prediction = model.predict(features)[0]

    st.success(f"🔋 **Predicted Charging Demand: {prediction:.2f}**")

    # Peak Demand Detection
    if prediction > np.percentile(df['demand'], 75):
        st.error("🚨 Peak Demand Expected!")
    else:
        st.success("✅ Normal Demand")

st.markdown("---")

# -------------------------------
# VISUALIZATION 1: DAILY DEMAND (Dynamic)
# -------------------------------
st.subheader("📊 Daily Demand Trend (Dynamic Graph)")
hours = list(range(24))
demo_demand = [model.predict([[h, day, month, dayofweek]])[0] for h in hours]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hours, demo_demand, marker='o', color='red')
ax.set_xlabel("Hour of Day (0-23)")
ax.set_ylabel("Predicted Demand")
ax.set_title(f"Predicted Demand vs Hour (Day {day}, Month {month})")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")

# -------------------------------
# VISUALIZATION 2: DEMAND BY HOUR
# -------------------------------
st.subheader("⏰ Demand Pattern by Hour (Historical Average)")

hourly_avg = df.groupby('hour')['demand'].mean()

fig2, ax2 = plt.subplots(figsize=(8, 4))
hourly_avg.plot(ax=ax2, marker='s', color='green')
ax2.set_xlabel("Hour")
ax2.set_ylabel("Average Demand")
ax2.grid(True)
st.pyplot(fig2)

st.markdown("---")

# -------------------------------
# MODEL PERFORMANCE ANALYSIS
# -------------------------------
st.subheader("📊 Model Performance Analysis")

# Sample data
df_sample = df.sample(100, random_state=42)
X_sample = df_sample[['hour','day','month','dayofweek']]
y_actual = df_sample['demand']
y_pred = model.predict(X_sample)

# Plot Actual vs Predicted
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.scatter(y_actual, y_pred, alpha=0.7, color='purple')
# Add a diagonal line to see perfect predictions easily
ax3.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
ax3.set_xlabel("Actual Demand")
ax3.set_ylabel("Predicted Demand")
ax3.set_title("Actual vs Predicted")
ax3.grid(True)
st.pyplot(fig3)

# Show Metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

st.write("### 📈 Model Metrics")
st.write(f"- **MAE:** {mae:.2f}")
st.write(f"- **RMSE:** {rmse:.2f}")
st.write(f"- **R² Score:** {r2:.2f}")
