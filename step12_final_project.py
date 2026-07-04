import pandas as pd
import joblib

# Load saved model
model = joblib.load("traffic_model.pkl")

print("=== Smart City Traffic Prediction System ===")

junction = int(input("Enter Junction Number: "))
id_value = int(input("Enter ID: "))

user_data = pd.DataFrame({
    "Junction": [junction],
    "ID": [id_value]
})

prediction = model.predict(user_data)

print("\nPredicted Traffic:", prediction[0])
print("Prediction completed successfully!")
