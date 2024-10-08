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
X = mall_df[["Annual Income (k$)", "Spending Score (1-100)"]]
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
mall_df["Cluster"] = kmeans.fit_predict(X)

# --- Visualization of clusters with descriptions ---
st.write("### Customer Segments with KMeans Clustering (Filtered Data)")
fig, ax = plt.subplots()
fig.patch.set_facecolor("#0414a0")  # Luminous blue background
ax.set_facecolor("#0414a0")
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
plt.title("Customer Segments with Descriptions", color="white")
plt.xlabel("Annual Income (k$)", color="white")
plt.ylabel("Spending Score (1-100)", color="white")
plt.legend()
st.pyplot(fig)

# Display cluster details
st.write("### Cluster Information (Filtered Data)")
cluster_info = filtered_df.groupby("Cluster").agg(
    avg_income=("Annual Income (k$)", "mean"),
    avg_spending=("Spending Score (1-100)", "mean"),
    count=("CustomerID", "count"),
)
st.write(cluster_info)

# User interaction: Select cluster
cluster_choice = st.sidebar.selectbox("Select Cluster to View Details", mall_df["Cluster"].unique())

# Display selected cluster details
st.write(f"### Selected Cluster {cluster_choice} Details")
st.write(filtered_df[filtered_df["Cluster"] == cluster_choice])

# --- Additional Visualizations ---

# 1. Spending Score Distribution by Gender
st.write(f"### Spending Score by Gender")
fig2, ax2 = plt.subplots()
fig2.patch.set_facecolor("#0414a0")  # Luminous blue background
ax2.set_facecolor("#0414a0")
sns.barplot(x="Genre", y="Spending Score (1-100)", data=filtered_df, palette="coolwarm", ax=ax2)
plt.title("Average Spending Score by Gender", color="white")
plt.xlabel("Gender", color="white")
plt.ylabel("Spending Score", color="white")
st.pyplot(fig2)

# 2. Income Distribution Across Clusters
st.write("### Income Distribution Across Clusters")
fig3, ax3 = plt.subplots()
fig3.patch.set_facecolor("#0414a0")  # Luminous blue background
ax3.set_facecolor("#0414a0")
sns.barplot(x="Cluster", y="Annual Income (k$)", data=filtered_df, palette="Blues", ax=ax3)
plt.title("Income Distribution in Clusters", color="white")
plt.xlabel("Cluster", color="white")
plt.ylabel("Annual Income (k$)", color="white")
st.pyplot(fig3)

# 3. Age vs Spending Score
st.write(f"### Age vs Spending Score (Colored by Clusters)")
fig4, ax4 = plt.subplots()
fig4.patch.set_facecolor("#0414a0")  # Luminous blue background
ax4.set_facecolor("#0414a0")
sns.scatterplot(x="Age", y="Spending Score (1-100)", hue="Cluster", data=filtered_df, palette="deep", ax=ax4)
plt.title("Age vs Spending Score", color="white")
plt.xlabel("Age", color="white")
plt.ylabel("Spending Score", color="white")
st.pyplot(fig4)

# Add a pie chart to visualize gender distribution in the clusters
st.write(f"### Gender Distribution in Cluster {cluster_choice}")
cluster_gender_dist = filtered_df[filtered_df["Cluster"] == cluster_choice]["Genre"].value_counts()
fig5, ax5 = plt.subplots()
fig5.patch.set_facecolor("#0414a0")  # Luminous blue background
ax5.set_facecolor("#0414a0")
cluster_gender_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax5, colors=['#66b3ff','#99ff99'])
plt.title(f"Gender Distribution in Cluster {cluster_choice}", color="white")
plt.ylabel('')
st.pyplot(fig5)

# Visualizing income vs. spending score with clusters
st.write("### Income vs Spending Score across Clusters")
fig6, ax6 = plt.subplots()
fig6.patch.set_facecolor("#0414a0")  # Luminous blue background
ax6.set_facecolor("#0414a0")
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", data=filtered_df, palette="deep", ax=ax6)
plt.title("Income vs Spending Score", color="white")
plt.xlabel("Annual Income (k$)", color="white")
plt.ylabel("Spending Score", color="white")
st.pyplot(fig6)
