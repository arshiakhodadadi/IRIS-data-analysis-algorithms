# IRIS-data-analysis-algorithms
The code showcases how these algorithms group data based on feature similarity without using class label

This Python script exemplifies the application of unsupervised machine learning techniques—specifically K-Means and Mean Shift clustering—on the Iris dataset using scikit-learn. The Iris dataset comprises 150 samples of iris flowers, each described by four features: sepal length, sepal width, petal length, and petal width, corresponding to three species: Setosa, Versicolor, and Virginica.
CodeProject
+2
ColabCodes
+2
C# Corner
+2

1. Data Loading and Visualization
The script begins by loading the Iris dataset and displaying the first ten samples along with their labels and feature names. An initial scatter plot visualizes the relationship between sepal length and sepal width, colored by species labels, providing an intuitive understanding of the data distribution.

2. Data Splitting and Normalization
The dataset is split into training and testing subsets using an 80-20 split. To ensure that each feature contributes equally to the clustering process, the features are standardized using StandardScaler, which scales the data to have a mean of zero and a standard deviation of one.

3. K-Means Clustering
K-Means clustering is applied to the standardized training data. To determine the optimal number of clusters (k), the script employs the Elbow Method, which involves computing the Within-Cluster Sum of Squares (WCSS) for different values of k and identifying the "elbow point" where the rate of decrease sharply changes. In this case, k=3 is chosen, aligning with the known number of species in the dataset.
CoderzColumn
+3
scikit-learn
+3
GitHub
+3

After fitting the K-Means model, the script predicts cluster labels for the training data and visualizes the clusters based on petal length and petal width, highlighting the cluster centers.

4. Model Evaluation
The trained K-Means model predicts labels for the test data. However, since K-Means is an unsupervised algorithm, the cluster labels are assigned arbitrarily and do not correspond directly to the true labels. Therefore, directly computing accuracy using accuracy_score may not provide meaningful insights without aligning the predicted labels with the true labels. A more appropriate evaluation would involve metrics like the Adjusted Rand Index or Normalized Mutual Information.

5. Mean Shift Clustering
The script also applies Mean Shift clustering, a non-parametric algorithm that does not require specifying the number of clusters beforehand. By setting a bandwidth parameter, Mean Shift identifies clusters by locating areas of high data density. The resulting clusters are visualized similarly to the K-Means clusters, providing a comparative perspective. 
GitHub

Applications in Machine Learning
This script serves as a practical demonstration of unsupervised learning techniques in machine learning. By applying clustering algorithms to the Iris dataset, it showcases how such methods can uncover inherent groupings within data without relying on labeled outcomes. These techniques are valuable in various applications, including customer segmentation, image compression, and pattern recognition.
