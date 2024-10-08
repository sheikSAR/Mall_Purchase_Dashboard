import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
mall_df = pd.read_csv("Mall_Customers.csv")

# Convert Gender to 0 and 1
mall_df['Genre'] = mall_df['Genre'].map({'Male': 0, 'Female': 1})

# Ensure correct data types
mall_df['Annual Income (k$)'] = pd.to_numeric(mall_df['Annual Income (k$)'], errors='coerce')
mall_df['Spending Score (1-100)'] = pd.to_numeric(mall_df['Spending Score (1-100)'], errors='coerce')
mall_df['Age'] = pd.to_numeric(mall_df['Age'], errors='coerce')
mall_df = mall_df.dropna()  # Drop rows with any missing values

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

# Map the new_gender input to 0 or 1
new_gender_numeric = 0 if new_gender == "Male" else 1

# Allow the user to submit new data
if st.sidebar.button("Add Customer"):
    new_data = {'CustomerID': [new_customer_id], 'Genre': [new_gender_numeric], 'Age': [new_age], 
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
st.write("### Customer Segments with KMeans Clustering (Filtered Data)")
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
st.pyplot(fig)

# --- Display cluster details ---
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

# Pie chart to visualize gender distribution in the selected cluster
st.write(f"### Gender Distribution in Cluster {cluster_choice}")
cluster_gender_dist = filtered_df[filtered_df["Cluster"] == cluster_choice]["Genre"].value_counts()
fig2, ax2 = plt.subplots()
cluster_gender_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['#66b3ff','#99ff99'])
plt.title(f"Gender Distribution in Cluster {cluster_choice}")
plt.ylabel('')
st.pyplot(fig2)
