import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# We open iris dataset and stock data and labels separetely
df = pd.read_csv("./iris.csv", sep=";", header=0, index_col=4)
header = df.columns.values
var_names = header[:4]
data = np.float32(df[header[:4]].values)
classes = df.index.values
colors = np.copy(classes)
# colors[colors=="setosa"]="red"
# colors[colors=="versicolor"]="green"
# colors[colors=="virginica"]="blue"
colors = ['r', 'g', 'b']

#Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)



#Visualize data with PCA
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(data)
data_pca = pca.transform(data)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(classes)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][classes==cl], data_pca[:,1][classes==cl], c=colors[i], label=cl, alpha=0.5)
ax.legend()
plt.title('PCA visualization')
# plt.show()

# We do k-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=300).fit(data_pca)
kmeans.score(data_pca)
prediction_pca = kmeans.predict(data_pca)  #predicted pca labels
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(prediction_pca)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][prediction_pca==cl], data_pca[:,1][prediction_pca==cl], c=colors[i], label=cl, alpha=0.5)
ax.legend()
plt.title('PCA Kmeans visualization')
# plt.show()

# We repeat kmeans several times, as the initialization is random, we may get different partitions
# for i in range(10):
#     kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=300).fit(data_pca)
#     kmeans.score(data_pca)
#     prediction_pca = kmeans.predict(data_pca)  # predicted pca labels
#


# We compute confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
classes_int = pd.Categorical(df.index).codes
conf_matrix = confusion_matrix(classes_int, prediction_pca)
fig, ax = plt.subplots()
df_cm = pd.DataFrame(conf_matrix, index=np.unique(prediction_pca), columns=np.unique(classes))
sns.heatmap(df_cm, annot=True)
plt.show()


#We compute silhouette score
silhouette_list = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(data_pca)
    silhouette_avg = silhouette_score(data_pca, cluster_labels)
    silhouette_list.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
fig, ax = plt.subplots()
ax.scatter(range(2, 11), silhouette_list)
ax.axvline(x=np.argmax(silhouette_list)+2, color="red", linestyle="--", label='the best partition')
ax.set_xlabel("Nb of clusters")
ax.set_ylabel("Silhouette score")
ax.legend()
plt.title("Silhouette score of different partitions")
plt.show()


# Question 9.
# We do the same thing as in previous questions, but we use raw iris data as input of clustering algorithm
# For the visualisation we still use pca_coordinates. For example
kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=300).fit(data)
kmeans.score(data)
prediction = kmeans.predict(data)  #predicted pca labels
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(prediction_pca)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][prediction==cl], data_pca[:,1][prediction==cl], c=colors[i], label=cl, alpha=0.5)
ax.legend()
plt.title('Kmeans visualization')
plt.show()


# Exercice B
#  needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet, inconsistent, maxRstat
from scipy.spatial.distance import pdist


np.random.seed(42)
a=np.random.multivariate_normal( [10, 0], [[3, 1], [1, 4]], size=[100, ])
b=np.random.multivariate_normal( [0, 20], [[3, 1], [1, 4]], size=[50, ])
X = np.concatenate((a, b),)
plt.scatter(X[:, 0], X[:, 1])
plt.title('My data distribution')
plt.show()


Z = linkage(X, 'ward', optimal_ordering=True)
c, coph_dists = cophenet(Z, pdist(X))
print ('Cophenetic Correlation : %1.2f' % c)
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (full)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
print(Z)

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

max_d = 14
clusters = fcluster(Z, max_d, criterion='distance')
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')


k=4
clusters = fcluster(Z, k, criterion='distance')
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')


# Exercice C
df_exo4_atm_extr = pd.read_csv('exo4_atm_extr.csv', sep=";")

X = df_exo4_atm_extr.drop(['Type'], axis=1)
Y = df_exo4_atm_extr['Type']
Y.columns = ['Type']
print(X.head())


from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

history_DB=[]
history_CH=[]
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42).fit(X)
    labels = kmeans.labels_
    value_DB = davies_bouldin_score(X, labels)
    value_CH = calinski_harabasz_score(X, labels)
    history_DB.append(value_DB)
    history_CH.append(value_CH)
x_value = np.arange(2,11)



fig, ax = plt.subplots()
ax.scatter(range(2, 11), history_DB)
ax.axvline(x=np.argmin(history_DB)+2, color="red", linestyle="--", label='the best partition')
ax.set_xlabel("Nb of clusters")
ax.set_ylabel("Davies-Bouldin score")
plt.show()

fig, ax = plt.subplots()
ax.scatter(range(2, 11), history_CH)
ax.axvline(x=np.argmax(history_CH)+2, color="red", linestyle="--", label='the best partition')
ax.set_xlabel("Nb of clusters")
ax.set_ylabel("Calinski-Harabasz score")
plt.show()
# Even if 10 clusters give the biggest CH score in absolute,
# we should consider that the most optimal nb of clusters is 4 as the gap between 3 and 4 is the biggest


# Question 4 for DB
#Visualize data with PCA
colors = "bgrcmykw"
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(X)
data_pca = pca.transform(X)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(Y)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][Y==cl], data_pca[:,1][Y==cl], c=colors[i], label=cl, alpha=0.5) #we generate new colors iteratively
ax.legend()
plt.title('PCA visualization')
plt.show()

# We do k-means for 4 clusters (minimum score of DB)
prediction = KMeans(n_clusters=4, random_state=42, n_init=5, max_iter=300).fit_predict(X)
fig, ax = plt.subplots()
for i, cl in enumerate(np.unique(prediction)): #we do the loop to be able to show labels
    ax.scatter(data_pca[:,0][prediction==cl], data_pca[:,1][prediction==cl], c=colors[i], label=cl, alpha=0.5) #we generate new colors iteratively
ax.legend()
plt.title('Kmeans visualization')
plt.show()
