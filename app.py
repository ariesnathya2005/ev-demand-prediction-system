from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to generate images
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = joblib.load("model.pkl")

def generate_visualization():
    df = pd.read_csv("ev_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    # Calculate average demand per hour
    hourly_demand = df.groupby('hour')['demand'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_demand.index, hourly_demand.values, color='b', marker='o')
    plt.title("Average EV Charging Demand vs Time (Hour of Day)")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Average Charging Demand")
    plt.grid(True)
    
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/plot.png")
    plt.close()

# Generate the plot once when the app starts
generate_visualization()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hour = int(request.form["hour"])
    day = int(request.form["day"])
    month = int(request.form["month"])

    features = np.array([[hour, day, month]])
    prediction = model.predict(features)

    return render_template("index.html", result=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
