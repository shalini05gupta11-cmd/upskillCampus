import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
train_data = pd.read_csv("train_aWnotuB.csv")

# Correlation of numeric columns
correlation = train_data.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(6, 4))
plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.show()