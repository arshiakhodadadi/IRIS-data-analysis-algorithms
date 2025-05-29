import numpy as np  
import matplotlib.pyplot as plt  
import sklearn.datasets as dt  
import sklearn.model_selection as ms  
import sklearn.preprocessing as pp  
import sklearn.cluster as cl  
from sklearn.metrics import accuracy_score  

# بارگذاری داده‌های آیریس  
iris_data = dt.load_iris()  

features = iris_data.data  
labels = iris_data.target  

feature_names = iris_data.feature_names  
target_names = iris_data.target_names  

print('Data Attributes (First 10 Samples):\n', features[:10],  
      '\nData Labels (First 10 Samples):\n', labels[:10],  
      '\nFeature Names:\n', feature_names,  
      '\nSpecies Names:\n', target_names)  

# مصور سازی داده‌ها  
plt.subplot(1,2,1)  
plt.scatter(features[:, 0], features[:, 1], s=15, c=labels, cmap='viridis')  
plt.xlabel('Sepal Length')  
plt.ylabel('Sepal Width')  
plt.title('Iris Data: Sepal Attributes')  

# تقسیم داده‌ها به مجموعه‌های آموزشی و تست  
train_features, test_features, train_labels, test_labels = ms.train_test_split(  
    features, labels, train_size=0.8, test_size=0.2, random_state=42  
)  

print(f'Training Features:\n{train_features}\nTesting Features:\n{test_features}\nTraining Labels:\n{train_labels}\nTesting Labels:\n{test_labels}')  

# نرمال‌سازی داده‌ها با StandardScaler  
scaler = pp.StandardScaler()  
train_features_scaled = scaler.fit_transform(train_features)  
test_features_scaled = scaler.transform(test_features)  

# استفاده از Elbow Method برای تعیین تعداد خوشه‌ها  
wcss = []  
for i in range(1, 11):  
    kmeans = cl.KMeans(n_clusters=i, random_state=42)  
    kmeans.fit(train_features_scaled)  
    wcss.append(kmeans.inertia_)  
    
plt.figure(figsize=(12, 5))  
plt.subplot(1, 2, 1)  
plt.plot(range(1, 11), wcss)  
plt.title('Elbow Method for Optimal k')  
plt.xlabel('Number of Clusters')  
plt.ylabel('WCSS')  # Within-cluster sum of squares  

# ایجاد مدل K-Means با تعداد بهینه خوشه‌ها  
optimal_k = 3  # با توجه به نمودار Elbow آن را تعیین کنید  
kmeans_model = cl.KMeans(n_clusters=optimal_k, random_state=42)  
kmeans_model.fit(train_features_scaled)  

# استخراج برچسب‌ها و مراکز خوشه‌ها  
predicted_labels = kmeans_model.labels_  
cluster_centers = kmeans_model.cluster_centers_  

# مصور سازی نتایج K-Means  
plt.subplot(1, 2, 2)  
plt.scatter(train_features_scaled[:, 2], train_features_scaled[:, 3], c=predicted_labels, s=15, cmap='viridis')  
plt.scatter(cluster_centers[:, 2], cluster_centers[:, 3], s=60, c='red', marker='o')  
plt.xlabel('Petal Length')  
plt.ylabel('Petal Width')  
plt.title('K-Means Clustering on Iris Dataset')  

# ارزیابی مدل K-Means  
test_predicted_labels = kmeans_model.predict(test_features_scaled)  
accuracy = accuracy_score(test_labels, test_predicted_labels)  
print(f'Accuracy of K-Means: {accuracy:.2f}')  

# اعمال Mean Shift  
MS = cl.MeanShift(bandwidth=0.85)  # اصلاح نام پارامتر  
MS.fit(train_features_scaled)  # استفاده از داده‌های نرمال‌شده  
mean_shift_labels = MS.labels_  

# مصور سازی نتایج Mean Shift  
plt.figure()  
plt.scatter(train_features_scaled[:, 2], train_features_scaled[:, 3], s=15, c=mean_shift_labels, cmap='viridis')  
plt.title('Mean Shift Clustering on Iris Dataset')  
plt.xlabel('Petal Length (scaled)')  
plt.ylabel('Petal Width (scaled)')  
plt.show()


