# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create a data object (dictionary)
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored': [20, 35, 50, 55, 65, 70, 75, 85, 90, 95]
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)
print("Data Frame:")
print(df)

# Step 3: Separate features (X) and target (Y)
X = df[['Hours_Studied']]   # Feature column
Y = df['Marks_Scored']      # Target column

# Step 4: Create Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X, Y)

# Step 6: Predict for a new value
new_data = pd.DataFrame({'Hours_Studied': [8.5]})
predicted_marks = model.predict(new_data)

# Step 7: Display results
print("\nSlope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print(f"Predicted Marks for {new_data['Hours_Studied'][0]} hours study: {predicted_marks[0]:.2f}")

# Step 8: Visualization
plt.scatter(df['Hours_Studied'], df['Marks_Scored'], color='blue', label='Actual Data')
plt.plot(df['Hours_Studied'], model.predict(X), color='red', label='Regression Line')
plt.scatter(new_data, predicted_marks, color='green', label='Predicted Point (8.5 hrs)')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Linear Regression: Hours Studied vs Marks Scored")
plt.legend()
plt.show()
