# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Step 1: Load the Iris dataset
iris = load_iris()

# Step 2: Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print("Data Frame Head:\n", df.head())
print("\nData Frame Info:\n")
print(df.info())

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Step 5: Add cluster labels to DataFrame
cluster_labels = kmeans.labels_
df['Cluster'] = cluster_labels

# Step 6: Display cluster centers
cluster_centers = kmeans.cluster_centers_
print("\nCluster Centers:\n", cluster_centers)

# Step 7: Evaluate Performance
true_labels = iris.target

# Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, cluster_labels)
print(f"\nAdjusted Rand Index (ARI): {ari}")

# Silhouette Score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")
