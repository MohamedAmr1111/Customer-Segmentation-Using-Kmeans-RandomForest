# Customer Segmentation Using K-Means and Random Forest

This project focuses on segmenting customers from a shopping mall dataset using K-Means clustering and Random Forest classification. The objective is to identify distinct customer segments based on their demographic information and shopping behavior, enabling better-targeted marketing strategies.
Project Structure

Data Loading and Exploration: The project begins with loading and exploring the dataset, checking for null values, and understanding the data distribution through descriptive statistics.

Data Visualization: Various visualizations are implemented to better understand the relationships between features, detect outliers, and observe the distribution of key variables.

Data Preprocessing: The Gender column is encoded using LabelEncoder, and features are selected for clustering.

Clustering Using K-Means: The Elbow Method is used to determine the optimal number of clusters, after which K-Means clustering is applied to segment the customers.

Classification Using Random Forest: After clustering, the dataset is split into training and testing sets, and a Random Forest classifier is trained to predict the clusters based on the selected features.

Model Evaluation: The accuracy of the Random Forest model is evaluated, and a detailed classification report is generated. A confusion matrix is also plotted to visualize the model's performance.

Feature Importance: The importance of each feature in predicting the clusters is calculated and displayed.

Variance Inflation Factor (VIF) Analysis: VIF is calculated to check for multicollinearity among features.

# Visualizations

Several visualizations are implemented to provide insights into the dataset and the results of the clustering and classification:

Heatmap of Feature Correlations:
A heatmap is used to visualize the correlations between different features in the dataset, helping to identify any strong relationships or potential multicollinearity.

Boxplots for Outlier Detection:
Boxplots are created for the numerical features (Age, Annual Income, and Spending Score) to detect and visualize any outliers in the data.

Histograms for Feature Distributions:
Histograms with KDE (Kernel Density Estimation) are plotted to show the distribution of the Age, Annual Income, and Spending Score features. Mean and median lines are added to provide a clearer understanding of the data distribution.

Pie Chart for Gender Distribution:
A pie chart is used to visualize the gender distribution in the dataset, offering a quick overview of the male-to-female ratio among the customers.

Elbow Method Plot:
The Elbow Method is visualized through a line plot showing the Within-Cluster Sum of Squares (WCSS) against the number of clusters, aiding in the selection of the optimal number of clusters for K-Means.

Confusion Matrix:
A confusion matrix is plotted to assess the performance of the Random Forest classifier, providing a visual representation of the correct and incorrect predictions made by the model.

Feature Importance Bar Plot:
A bar plot is used to display the importance of each feature in the Random Forest model, helping to identify which features contribute most to predicting customer segments.

# Usage

 Data: Ensure the dataset (Shopping Mall Customer Segmentation Data.csv) is in the same directory as the script.

Execution: Run the script to perform customer segmentation and visualize the results.

    Dependencies:
        Python 3.x
        pandas
        numpy
        seaborn
        matplotlib
        scikit-learn
        statsmodels

    Output: The script will output the clustered data, model accuracy, and various plots to help understand the segmentation.

# Conclusion
This project demonstrates the application of unsupervised (K-Means) and supervised (Random Forest) learning algorithms to segment customers. The visualizations play a crucial role in understanding the dataset and evaluating the model's performance, making the analysis more interpretable and actionable.
