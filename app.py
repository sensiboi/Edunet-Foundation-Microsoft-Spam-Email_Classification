import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
#import dataset
data = pd.read_csv("spambase.csv")
data.head()
data.columns
data.dtypes
data.columns.isnull()
grouped_data = data.groupby(data['class'])
grouped_data[data.columns[0]].describe(include='all').transpose()
print('index','  ','0s','     ','other', '     columns')
for i in range(len(data.columns)):
    c = data[data[data.columns[i]] == 0].shape[0]
    print (i,'    ',c,'     ',4601-c,'   ', data.columns[i])
# Plot box plot
ax = sns.boxplot(x = 'class', y = 'word_freq_our', data = data, whis = 10)
ax.set(title = 'Box plot ', xlabel = 'Class', ylabel = 'word_freq_our')
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
for i in range(0,6):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[i], whis=10)
plt.tight_layout()fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(6,12):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(12,18):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(18,24):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(24,30):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(30,36):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(36,42):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(42,48):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(48,54):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
fig, axes = plt.subplots(1, 6, figsize = (10, 5))
j = 0
for i in range(54,56):
    sns.boxplot(x='class', y = data.columns[i], data = data, orient = 'v', ax = axes[j], whis=10)
    j+=1
plt.tight_layout()
#Splitting dataset for classification
#creating test and train sets
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data['class'],test_size=0.2, random_state=0)
#x_test = x_test.iloc[:,:-1]

x_train.head()
#creating validation set
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25, random_state=0)
x_val.count()
x_train.head()
print(x_train.shape)
print(x_val.shape)
print(x_te#standardizing datasets for classification
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled)
x_val_scaled = scaler.transform(x_val)
x_val_scaled = pd.DataFrame(x_val_scaled)st.shape)
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(x_train_scaled,y_train)
print(classifier.tree_.__getstate__()['nodes'])
len(classifier.tree_.__getstate__()['nodes'])
conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
cols = [0,1,2,4,5,6,7,8,9,10,11,12,15,16,17,20,22,23,24,36,44,49,51,52,56]
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(x_train_scaled.iloc[:,cols],y_train)
# structure of the decision tree classifier
print(classifier.tree_.__getstate__()['nodes'])
len(classifier.tree_.__getstate__()['nodes'])
 y_pred = classifier.predict(x_val_scaled.iloc[:,cols])
accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score
conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
classifier = KNeighborsClassifier(n_neighbors = 3, weights='distance')
classifier.fit(x_train_scaled,y_train)
y_pred = classifier.predict(x_val_scaled)
conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
cols = [0,1,2,5,6,8,11,16,18,20,22,23,24,49,51,56]
classifier = KNeighborsClassifier(n_neighbors = 5, weights='distance')
classifier.fit(x_train_scaled.iloc[:,cols],y_train)
#selecting optimal value of K
cols = [0,1,2,5,6,8,11,16,18,20,22,23,24,49,51,56]
for i in range(1,20):
    classifier = KNeighborsClassifier(n_neighbors = i,weights='distance')
    classifier.fit(x_train_scaled.iloc[:,cols],y_train)
    y_pred = classifier.predict(x_val_scaled.iloc[:,cols])
    accuracy = metrics.accuracy_score(y_val,y_pred)
    error = 1 - accuracy
    precision = metrics.precision_score(y_val, y_pred, average = None)
    recall = metrics.recall_score(y_val, y_pred, average = None)
    F1_score = metrics.f1_score(y_val, y_pred, average = None)
    print([i,accuracy, error, precision, recall, F1_score])
    y_pred = classifier.predict(x_val_scaled.iloc[:,cols])
accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
     conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
classifier = GaussianNB()
classifier.fit(x_train_scaled,y_train)
y_pred = classifier.predict(x_val_scaled)
conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
cols = [0,1,2,4,6,8,11,15,16,17,20,23,24,36,44,49,52,56]
classifier = GaussianNB()
classifier.fit(x_train_scaled.iloc[:,cols],y_train)
y_pred = classifier.predict(x_val_scaled.iloc[:,cols])

     accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
classifier = SVC(kernel = 'rbf', gamma = 'auto')
classifier.fit(x_train_scaled,y_train) 
y_pred = classifier.predict(x_val_scaled)
conf_matrix = metrics.confusion_matrix(y_val,y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".2f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix) accuracy = metrics.accuracy_score(y_val,y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, y_pred, average = None)
recall = metrics.recall_score(y_val, y_pred, average = None)
F1_score = metrics.f1_score(y_val, y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
SVM_cols = [0,4,6,7,8,9,11,15,16,17,20,22,23,24,36,52,56]
SVM_classifier = SVC(kernel = 'rbf', gamma = 'auto')
SVM_classifier.fit(x_train_scaled.iloc[:,SVM_cols],y_train)
SVM_y_pred = SVM_classifier.predict(x_val_scaled.iloc[:,SVM_cols])
accuracy = metrics.accuracy_score(y_val,SVM_y_pred)
error = 1 - accuracy
precision = metrics.precision_score(y_val, SVM_y_pred, average = None)
recall = metrics.recall_score(y_val, SVM_y_pred, average = None)
F1_score = metrics.f1_score(y_val, SVM_y_pred, average = None)
print("Accuracy = ",accuracy, "\nError = ",error, "\nPrecision =",precision,"\nRecall = ", recall,"\nF1 Score = ", F1_score)
conf_matrix = metrics.confusion_matrix(y_val,SVM_y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')
plt.tight_layout()
print(conf_matrix)
#Splitting dataset for clustering
data_x = data.iloc[:,:-1]
#data_x.head()
data_y = data['class']
#data_y.head()
#standardizing datasets for clustering
scaler = StandardScaler()
scaler.fit(data_x)
data_x_scaled = scaler.transform(data_x)
data_x_scaled = pd.DataFrame(data_x_scaled)
clustering = KMeans(n_clusters = 2,init = 'random',n_init = 10, random_state = 0).fit(data_x_scaled)
clusters = clustering.labels_
print("Clusters are : ",np.unique(clusters))
print("Number of clusters = ",len(clusters))
adjusted_rand_index = metrics.adjusted_rand_score(data_y,clusters)
silhouette_coefficient = metrics.silhouette_score(data_x_scaled,clusters)
print("Adjusted Random Index =",adjusted_rand_index, "\nSilhouette Coefficient = ",silhouette_coefficient)
cont_matrix = metrics.cluster.contingency_matrix(clusters, data_y)
sns.heatmap(cont_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Contingency matrix')
plt.tight_layout()
print(cont_matrix)
#cols = [0,1,2,4,5,6,7,8,9,10,11,12,15,16,17,18,20,22,23,24,36,44,49,51,52,56]
#cols = [0,1,2,4,5,6,7,8,9,10,11,12,15,16,17,20,22,23,24,36,44,49,51,52,56]
#cols = [4,5,6,8,16,18,20,23,24,51,56]
#cols = [0,1,2,4,6,8,11,15,16,17,20,23,24,36,44,49,52,56]
Kmeans_cols = [4,6,7,8,9,15,20,22,23,24,36,52,56]
clustering = KMeans(n_clusters = 2,init = 'random',n_init = 10, random_state = 0).fit(data_x_scaled.iloc[:,Kmeans_cols])
clusters = clustering.labels_
print("Clusters are : ",np.unique(clusters))
print("Number of clusters = ",len(clusters))
adjusted_rand_index = metrics.adjusted_rand_score(data_y,clusters)
silhouette_coefficient = metrics.silhouette_score(data_x_scaled.iloc[:,Kmeans_cols],clusters)
print("Adjusted Random Index =",adjusted_rand_index, "\nSilhouette Coefficient = ",silhouette_coefficient)
cont_matrix = metrics.cluster.contingency_matrix(clusters, data_y)
sns.heatmap(cont_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Contingency matrix')
plt.tight_layout()
print(cont_matrix)
