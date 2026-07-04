import pandas as pd

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Convert DateTime to datetime format
train_data["DateTime"] = pd.to_datetime(train_data["DateTime"])

# Create new features
train_data["Year"] = train_data["DateTime"].dt.year
train_data["Month"] = train_data["DateTime"].dt.month
train_data["Day"] = train_data["DateTime"].dt.day
train_data["Hour"] = train_data["DateTime"].dt.hour

# Show first 5 rows
print(train_data.head())