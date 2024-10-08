import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset (ensure the dataset is in the same directory or provide the correct path)
mall_df = pd.read_csv(
    "your_dataset.csv"
)  # Replace with the correct path or upload the dataset separately

# Sidebar for user input
st.sidebar.header("User Input Features")

# Display the dataset
st.write("### Mall Customer Segmentation Data")
st.dataframe(mall_df.head())

# KMeans Clustering
X = mall_df[["Annual Income (k$)", "Spending Score (1-100)"]]
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
mall_df["Cluster"] = kmeans.fit_predict(X)

# Visualization of clusters
st.write("### Customer Segments with KMeans Clustering")
fig, ax = plt.subplots()
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    data=mall_df,
    palette="viridis",
    ax=ax,
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="red",
    label="Centroids",
)
st.pyplot(fig)

# Display cluster details
st.write("### Cluster Information")
cluster_info = mall_df.groupby("Cluster").agg(
    avg_income=("Annual Income (k$)", "mean"),
    avg_spending=("Spending Score (1-100)", "mean"),
    count=("CustomerID", "count"),
)
st.write(cluster_info)

# User interaction: Select cluster
cluster_choice = st.sidebar.selectbox(
    "Select Cluster to View Details", mall_df["Cluster"].unique()
)

# Display selected cluster details
st.write(f"### Selected Cluster {cluster_choice} Details")
st.write(mall_df[mall_df["Cluster"] == cluster_choice])
