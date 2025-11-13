# Importing libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 1: Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("---- Iris Dataset (first 5 rows) ----")
print(df.head())

# Step 2: Define features (X) and target (Y)
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
Y = df['petal width (cm)']

# Step 3: Create and train model
model = LinearRegression()
model.fit(X, Y)

# Step 4: Predictions
Y_pred = model.predict(X)

# Step 5: Evaluate performance using metrics
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)

# Step 6: Display model coefficients and performance
print("\n---- Model Coefficients ----")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print("Intercept:", model.intercept_)

print("\n---- Performance Metrics ----")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# Step 7: Predict for a new sample
new_data = pd.DataFrame({
    'sepal length (cm)': [6.0],
    'sepal width (cm)': [3.0],
    'petal length (cm)': [4.5]
})
predicted_width = model.predict(new_data)
print(f"\nPredicted Petal Width for {new_data.values.tolist()[0]}: {predicted_width[0]:.2f}")
