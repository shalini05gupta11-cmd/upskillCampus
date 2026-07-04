import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Features and target
X = train_data[["Junction", "ID"]]
y = train_data["Vehicles"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "traffic_model.pkl")

print("Model saved successfully as traffic_model.pkl")