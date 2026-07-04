import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Features and target
X = train_data[["Junction", "ID"]]
y = train_data["Vehicles"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")

# Future data (example)
future_data = pd.DataFrame({
    "Junction": [1, 2, 3],
    "ID": [2017110101, 2017110102, 2017110103]
})

# Predict future traffic
future_predictions = model.predict(future_data)

print("\nFuture Traffic Predictions:")
print(future_predictions)