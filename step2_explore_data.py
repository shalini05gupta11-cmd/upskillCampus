import pandas as pd

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Dataset information
print("Dataset Shape:", train_data.shape)

print("\nColumns:")
print(train_data.columns)

print("\nMissing Values:")
print(train_data.isnull().sum())

print("\nStatistics:")
print(train_data.describe())

print("\nFirst 5 Rows:")
print(train_data.head())