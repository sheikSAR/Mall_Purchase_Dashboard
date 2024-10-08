import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
mall_df = pd.read_csv("Mall_Customers.csv")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Allow the user to filter by age range
age_range = st.sidebar.slider(
    "Select Age Range", min_value=int(mall_df.Age.min()), max_value=int(mall_df.Age.max()), value=(18, 70)
)
filtered_df = mall_df[(mall_df.Age >= age_range[0]) & (mall_df.Age <= age_range[1])]

# Display the dataset
st.write(f"### Mall Customer Segmentation Data (Age Filter: {age_range[0]} - {age_range[1]})")
st.dataframe(filtered_df.head())

# KMeans Clustering
X = filtered_df[["Annual Income (k$)", "Spending Score (1-100)"]]
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
filtered_df["Cluster"] = kmeans.fit_predict(X)

# Visualization of clusters
st.write("### Customer Segments with KMeans Clustering")
fig, ax = plt.subplots()
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    data=filtered_df,
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
plt.title("Customer Segments based on Annual Income and Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
st.pyplot(fig)

# Add bar chart to visualize gender distribution in clusters
st.write("### Gender Distribution per Cluster")
gender_counts = filtered_df.groupby("Cluster")["Genre"].value_counts().unstack()
st.bar_chart(gender_counts)

# Display cluster details
st.write("### Cluster Information")
cluster_info = filtered_df.groupby("Cluster").agg(
    avg_income=("Annual Income (k$)", "mean"),
    avg_spending=("Spending Score (1-100)", "mean"),
    count=("CustomerID", "count"),
)
st.write(cluster_info)

# User interaction: Select cluster
cluster_choice = st.sidebar.selectbox("Select Cluster to View Details", filtered_df["Cluster"].unique())

# Display selected cluster details
st.write(f"### Selected Cluster {cluster_choice} Details")
st.write(filtered_df[filtered_df["Cluster"] == cluster_choice])

# Add pie chart to show age distribution in the selected cluster
st.write(f"### Age Distribution in Cluster {cluster_choice}")
cluster_age_dist = filtered_df[filtered_df["Cluster"] == cluster_choice]["Age"]
fig2, ax2 = plt.subplots()
cluster_age_dist.plot(kind='hist', bins=10, alpha=0.7, ax=ax2)
plt.title(f"Age Distribution in Cluster {cluster_choice}")
st.pyplot(fig2)
