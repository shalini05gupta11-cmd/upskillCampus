import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Convert DateTime column
train_data["DateTime"] = pd.to_datetime(train_data["DateTime"])

# Plot traffic over time
plt.figure(figsize=(12,5))
plt.plot(train_data["DateTime"], train_data["Vehicles"])
plt.title("Traffic Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Vehicles")
plt.show()