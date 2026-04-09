# ⚡ EV Charging Demand Prediction System

A full-stack Machine Learning web application built to predict Electric Vehicle (EV) charging demand based on temporal features. The project uses a Random Forest Regressor to provide accurate, real-time load predictions and features a highly customized, responsive Streamlit dashboard.

## 🌟 Key Features

* **Interactive Prediction Dashboard:** Custom-styled dark UI using Streamlit, featuring real-time sliders for Hour, Day, and Month.
* **Peak Demand Warnings:** Intelligent threshold detection changes UI elements dynamically (High/Medium/Low demand) to alert operators.
* **Comprehensive Analytics:**
  * **Hourly Forecast Chart:** Visualizes expected demand throughout the day based on selected features.
  * **Weekly Pattern Analysis:** Tracks consumption differences across the week.
  * **Feature Importance:** Ranks the impact of variables like Time of Day vs. Day of Week.
* **Model Evaluation Metrics:** Displays MAE, RMSE, and R² Score evaluated on simulated hold-out test sets, alongside residual distribution histograms.

## 🛠️ Technology Stack

* **Frontend & Backend UI:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (RandomForestRegressor)
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Deployment:** Render

## 🚀 Quick Start (Local Setup)

Follow these steps to run the application on your local machine.

### 1. Clone the repository
```bash
git clone https://github.com/ariesnathya2005/ev-demand-prediction-system.git
cd ev-demand-prediction-system
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate Data & Train the Model
Run the setup script to generate synthetic EV history data and train the `model.pkl` binary:
```bash
python train_model.py
```

### 4. Run the Dashboard
Launch the custom Streamlit application:
```bash
streamlit run app.py
```
The app will automatically open in your browser at `http://localhost:8501`.

## 🌐 Cloud Deployment

This application is fully compatible with PaaS providers like **Render**.

**Live Demo:** [https://ev-demand-prediction-system.onrender.com](https://ev-demand-prediction-system.onrender.com)

**Deployment Steps (Render):**
1. Connect your GitHub repository to Render.
2. Select **Web Service**.
3. Set the Build Command: `pip install -r requirements.txt`
4. Set the Start Command: `streamlit run app.py --server.port $PORT`
5. Deploy!

## 📂 Project Structure

```text
├── app.py               # Main Streamlit web application & routing
├── train_model.py       # Data generation and ML model training script
├── ev_dashboard.py      # Custom UI design layout templates
├── requirements.txt     # Python package dependencies
├── model.pkl            # Serialized Random Forest model (generated)
└── README.md            # Project documentation
```

