import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load Dataset
# Note: Update "ev_data.csv" with the actual path to your dataset
df = pd.read_csv("ev_data.csv")
print(df.head())

# Clean Data
df = df.dropna()

# Convert Timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Select Features
X = df[['hour', 'day', 'month']]
y = df['demand']   # or energy/session column

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate Model
pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, pred)}")

# Save Model
joblib.dump(model, "model.pkl")
