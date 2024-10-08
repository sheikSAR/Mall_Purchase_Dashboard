import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
mall_df = pd.read_csv("Mall_Customers.csv")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Allow the user to input a new customer's data
st.sidebar.write("## Add New Customer Data")

# Auto-increment CustomerID by taking the max ID in the dataset
new_customer_id = mall_df['CustomerID'].max() + 1
new_gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
new_age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=25)
new_income = st.sidebar.number_input("Annual Income (k$)", min_value=15, max_value=140, value=50)
new_spending_score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Allow the user to submit new data
if st.sidebar.button("Add Customer"):
    new_data = {'CustomerID': [new_customer_id], 'Genre': [new_gender], 'Age': [new_age], 
                'Annual Income (k$)': [new_income], 'Spending Score (1-100)': [new_spending_score]}
    new_customer_df = pd.DataFrame(new_data)
    mall_df = pd.concat([mall_df, new_customer_df], ignore_index=True)

# --- Filter options ---
st.sidebar.write("## Filter Data")
age_range = st.sidebar.slider("Select Age Range", min_value=int(mall_df["Age"].min()), 
                              max_value=int(mall_df["Age"].max()), value=(18, 70))
income_range = st.sidebar.slider("Select Income Range (k$)", min_value=int(mall_df["Annual Income (k$)"].min()), 
                                 max_value=int(mall_df["Annual Income (k$)"].max()), value=(15, 140))
spending_range = st.sidebar.slider("Select Spending Score Range (1-100)", 
                                   min_value=int(mall_df["Spending Score (1-100)"].min()), 
                                   max_value=int(mall_df["Spending Score (1-100)"].max()), value=(1, 100))

# Apply filters to the dataframe
filtered_df = mall_df[(mall_df["Age"].between(age_range[0], age_range[1])) &
                      (mall_df["Annual Income (k$)"].between(income_range[0], income_range[1])) &
                      (mall_df["Spending Score (1-100)"].between(spending_range[0], spending_range[1]))]

# KMeans Clustering
X = filtered_df[["Annual Income (k$)", "Spending Score (1-100)"]]
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
filtered_df["Cluster"] = kmeans.fit_predict(X)

# --- Visualization of clusters with descriptions ---
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

# Add cluster descriptions to the plot
descriptions = ['Low-income, low-spending', 'Medium-income, medium-spending', 
                'High-income, low-spending', 'High-income, high-spending', 
                'Low-income, high-spending']
centroids = kmeans.cluster_centers_

# Adding descriptions on the plot near the centroids
for i, desc in enumerate(descriptions):
    plt.text(centroids[i, 0], centroids[i, 1] + 2, desc, fontsize=10, ha='center', color='black', bbox=dict(facecolor='white', alpha=0.8))

plt.title("Customer Segments with Descriptions")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
st.pyplot(fig)

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

# --- Additional Visualizations ---

# Add a pie chart to visualize gender distribution in the clusters
st.write(f"### Gender Distribution in Cluster {cluster_choice}")
cluster_gender_dist = filtered_df[filtered_df["Cluster"] == cluster_choice]["Genre"].value_counts()
fig2, ax2 = plt.subplots()
cluster_gender_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['#66b3ff','#99ff99'])
plt.title(f"Gender Distribution in Cluster {cluster_choice}")
plt.ylabel('')
st.pyplot(fig2)

# Add bar chart for Age distribution in selected cluster
st.write(f"### Age Distribution in Cluster {cluster_choice}")
fig3, ax3 = plt.subplots()
sns.histplot(filtered_df[filtered_df["Cluster"] == cluster_choice]["Age"], bins=10, kde=True, ax=ax3)
plt.title(f"Age Distribution in Cluster {cluster_choice}")
st.pyplot(fig3)

# Visualizing income vs. spending score with clusters
st.write("### Income vs Spending Score across Clusters")
fig4, ax4 = plt.subplots()
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", data=filtered_df, palette="deep", ax=ax4)
plt.title("Income vs Spending Score")
st.pyplot(fig4)
