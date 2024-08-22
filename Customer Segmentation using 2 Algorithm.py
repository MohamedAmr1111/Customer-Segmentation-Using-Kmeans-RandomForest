# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 22:12:38 2024

@author: Odoo
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('Shopping Mall Customer Segmentation Data .csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Data column names
headers = data.columns.values
print("Headers:", headers)

# Check the relations between features
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.show()

# Check for null values
print("Null values:\n", data.isnull().sum())

# Check for outliers through boxplot
numerical_cols = ['Age', 'Annual Income', 'Spending Score']

for col in numerical_cols:
    sns.boxplot(data=data, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()

# Determine number of bins for histogram
n = len(data['Age'])    
num_bins = int(np.ceil(np.log2(n) + 1)) 

# Visualize the distribution
for col in numerical_cols:
    sns.histplot(data, x=col, kde=True, bins=num_bins)
    plt.axvline(data[col].mean(), color='red', linestyle='--')
    plt.axvline(data[col].median(), color='green', linestyle='-')
    plt.title(f'Distribution of {col}')
    plt.show()

# Creating a pie chart for gender distribution
gender_counts = data['Gender'].value_counts()
gender_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"), shadow=True, startangle=90)
plt.title('Gender Distribution')
plt.ylabel('')  # Hide the y-label for a cleaner look
plt.show()

# Maximum and Minimum Spending Scores
max_score = data[data['Spending Score'] == data['Spending Score'].max()]
min_score = data[data['Spending Score'] == data['Spending Score'].min()]    
print("Max Spending Score:\n", max_score)
print("Min Spending Score:\n", min_score)

# Encode the 'Gender' column
label_encoder = LabelEncoder()
# MALE = 1 , FEMALE = 0
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Select features for clustering
features = data[['Age', 'Gender', 'Annual Income', 'Spending Score']]
print("Features shape:", features.shape)

# Apply the Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)  # WCSS for the current k

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with the optimal number of clusters (let's assume k=3 from the elbow method)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)
#data = data.drop('Customer ID', axis=1)
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data['Cluster'], test_size=0.3, random_state=42)

# Initialize and train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed classification report
print(classification_report(y_test, y_pred))

# Optionally, visualize the results (e.g., a confusion matrix)
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = classifier.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': importances
})

# Sort the DataFrame by importance (optional)
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importances)

new_data = [[30,1, 151479, 89]]  # Example new data point
cluster_prediction = kmeans.predict(new_data)
print(f"New data point belongs to cluster: {cluster_prediction}")



new_data = [[30,1, 151479, 89]]  # Example new data point
cluster_prediction = classifier.predict(new_data)
print(f"New data point belongs to cluster: {cluster_prediction}")



from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame() 
vif_data["feature"] = features.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(features.values, i) 
                          for i in range(len(features.columns))] 
print(vif_data)










