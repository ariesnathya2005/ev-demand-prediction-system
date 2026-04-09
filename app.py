import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("⚡ EV Charging Demand Prediction System")
st.write("Predict EV charging demand based on time inputs")

# -------------------------------
# USER INPUT
# -------------------------------
st.write("### Input Parameters")
hour = st.slider("Select Hour", 0, 23, 12)
day = st.slider("Select Day", 1, 31, 15)
month = st.slider("Select Month", 1, 12, 6)
dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Demand"):
    # (Fix applied: Our dataset does not use station_id, so we only pass 4 features)
    features = np.array([[hour, day, month, dayofweek]])
    
    prediction = model.predict(features)[0]

    st.success(f"🔋 Predicted Charging Demand: {prediction:.2f}")

    # BONUS: Added Extra Marks feature
    # (Note: Set threshold to 100000 because your dataset's demand is in the ~50k-120k range)
    if prediction > 100000:
        st.warning("⚠️ High demand expected! Peak time.")
    else:
        st.info("✅ Normal demand")

st.markdown("---")

# -------------------------------
# SIMPLE VISUALIZATION
# -------------------------------
st.subheader("📊 Daily Demand Trend (Dynamic Graph)")
st.write(f"Showing predictions for all 24 hours on Day: {day}, Month: {month}, Day of Week: {dayofweek}")

# Generate demo data for visualization based on slide inputs
hours = list(range(24))
demo_demand = [model.predict([[h, day, month, dayofweek]])[0] for h in hours]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours, demo_demand, marker='o', color='red')
ax.set_xlabel("Hour of Day (0-23)")
ax.set_ylabel("Predicted Demand")
ax.set_title("Predicted Demand vs Hour")
ax.grid(True)

st.pyplot(fig)
