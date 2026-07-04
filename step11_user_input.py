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

# Take user input
junction = int(input("Enter Junction Number: "))
id_value = int(input("Enter ID: "))

# Create input data
user_data = pd.DataFrame({
    "Junction": [junction],
    "ID": [id_value]
})

# Predict
prediction = model.predict(user_data)

print("Predicted Traffic:", prediction[0])