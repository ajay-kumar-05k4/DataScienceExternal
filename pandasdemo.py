import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 28],
    'Salary': [50000, 60000, 70000, 80000, 65000]
}

df = pd.DataFrame(data)

print("---- Dataset ----")
print(df)

print("\n---- Head ----")
print(df.head())

print("\n---- Tail ----")
print(df.tail())

print("\n---- Info ----")
print(df.info())

print("\n---- Describe ----")
print(df.describe())

print("\n---- Mean ----")
print(df[['Age', 'Salary']].mean())

print("\n---- Median ----")
print(df[['Age', 'Salary']].median())

print("\n---- Mode ----")
print(df[['Age', 'Salary']].mode())

plt.scatter(df['Age'], df['Salary'], color='blue', marker='o')
plt.title('Scatter Plot - Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(True)
plt.show()
