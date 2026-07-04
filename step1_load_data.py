import pandas as pd

# Load the datasets
train_data = pd.read_csv("train_aWnotuB.csv")
test_data = pd.read_csv("datasets_8494_11879_test_BdBKkAj.csv")

# Display basic information
print("Training Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

print("\nFirst 5 rows of Training Data:")
print(train_data.head())