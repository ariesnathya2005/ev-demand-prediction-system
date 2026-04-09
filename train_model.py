import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# STEP 1: LOAD DATASET
# -------------------------------
df = pd.read_csv("ev_data.csv")

print("Columns in dataset:", df.columns)

# -------------------------------
# STEP 2: HANDLE TIMESTAMP
# -------------------------------
# Try different possible column names
time_col = None
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        time_col = col
        break

if time_col is None:
    raise Exception("No timestamp column found!")

df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

# Remove invalid timestamps
df = df.dropna(subset=[time_col])

# Extract features
df["hour"] = df[time_col].dt.hour
df["day"] = df[time_col].dt.day
df["month"] = df[time_col].dt.month
df["dayofweek"] = df[time_col].dt.dayofweek

# -------------------------------
# STEP 3: HANDLE TARGET COLUMN
# -------------------------------
target_col = None

possible_targets = ["energy", "kwh", "demand", "sessions"]

for col in df.columns:
    for key in possible_targets:
        if key in col.lower():
            target_col = col
            break

if target_col is None:
    raise Exception("No target column found!")

print("Using target column:", target_col)

# Remove missing target values
df = df.dropna(subset=[target_col])

# -------------------------------
# STEP 4: HANDLE LOCATION (OPTIONAL)
# -------------------------------
if "station_id" in df.columns:
    df["station_id"] = df["station_id"].astype("category").cat.codes
    features = ["hour", "day", "month", "dayofweek", "station_id"]
else:
    features = ["hour", "day", "month", "dayofweek"]

# -------------------------------
# STEP 5: DEFINE X and y
# -------------------------------
X = df[features]
y = df[target_col]

# -------------------------------
# STEP 6: TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 7: TRAIN MODEL
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# STEP 8: EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------------
# STEP 9: SAVE MODEL
# -------------------------------
joblib.dump(model, "model.pkl")

print("\n✅ Model saved as model.pkl")
