# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Step 1: Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = pd.Series(iris.target, name='Species')

print("First 5 rows of the dataset:")
print(X.head())

# Step 2: Split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 3: Create and train KNN model
k = 5  # You can change k here
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)

# Step 4: Evaluate model
Y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred, target_names=iris.target_names)

print(f"\nModel Evaluation for k = {k}:")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix - k = {k}")
plt.show()

# Step 5: Take input from user
print("\nEnter the following features of the iris flower to predict its species:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)

# Step 6: Predict species
predicted_class_index = knn.predict(user_input)[0]
predicted_species = iris.target_names[predicted_class_index]

print(f"\nThe predicted species is: {predicted_species}")
