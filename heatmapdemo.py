import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Age': [25, 30, 35, 40, 28],
    'Salary': [50000, 60000, 70000, 80000, 65000],
    'Experience': [1, 3, 5, 7, 2],
    'Score': [80, 85, 90, 95, 88]
}

df = pd.DataFrame(data)

print("---- Dataset ----")
print(df)

# Step 1: Compute correlation matrix
corr = df.corr()

print("\n---- Correlation Matrix ----")
print(corr)

# Step 2: Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap - Correlation Between Variables")
plt.show()
