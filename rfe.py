from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Create Logistic Regression model
model = LogisticRegression(max_iter=200)

# Apply RFE to select 2 best features
rfe = RFE(model, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)

# Print selected features
print("Selected Features:")
for name, selected in zip(feature_names, rfe.support_):
    if selected:
        print(name)

# Plot before and after RFE
plt.figure(figsize=(12, 5))

# Before RFE
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.title("Before RFE (All Features)")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])

# After RFE
plt.subplot(1, 2, 2)
plt.scatter(X_rfe[:, 0], X_rfe[:, 1], c=y, cmap='rainbow')
plt.title("After RFE (Selected Features)")
plt.xlabel("petal length")
plt.ylabel("petal width")

plt.show()