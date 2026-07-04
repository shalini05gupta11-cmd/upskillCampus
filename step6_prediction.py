import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Convert DateTime column
train_data["DateTime"] = pd.to_datetime(train_data["DateTime"])

# Create new features
train_data["Year"] = train_data["DateTime"].dt.year
train_data["Month"] = train_data["DateTime"].dt.month
train_data["Day"] = train_data["DateTime"].dt.day
train_data["Hour"] = train_data["DateTime"].dt.hour

# Features and target
X = train_data[["Junction", "Year", "Month", "Day", "Hour"]]
y = train_data["Vehicles"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Predict
predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(predictions[:10])