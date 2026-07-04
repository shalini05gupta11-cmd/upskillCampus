import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Average vehicles at each junction
avg_traffic = train_data.groupby("Junction")["Vehicles"].mean()

# Bar chart
plt.figure(figsize=(6,4))
avg_traffic.plot(kind="bar")

plt.title("Average Traffic at Each Junction")
plt.xlabel("Junction")
plt.ylabel("Average Number of Vehicles")

plt.tight_layout()
plt.show()