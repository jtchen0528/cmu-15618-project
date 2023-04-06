# Apply DBSCAN on PCA-reduced MNIST

# Conda Environment
# conda create -n tf tensorflow
# conda activate tf
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge matplotlib

useLocal = 1

from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def get_neighbors(X, i, eps):
    neighbors = []
    for j in range(len(X)):
        if i != j and euclidean_distance(X[i], X[j]) <= eps:
            neighbors.append(j)
    return neighbors

def expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples):
    labels[i] = cluster_id
    j = 0
    while j < len(neighbors):
        neighbor = neighbors[j]
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_id
        elif labels[neighbor] == 0:
            labels[neighbor] = cluster_id
            new_neighbors = get_neighbors(X, neighbor, eps)
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors
        j += 1

def dbscan(X, eps, min_samples):
    labels = np.zeros(len(X))
    cluster_id = 0
    
    for i in range(len(X)):
        if labels[i] != 0:
            continue
        
        neighbors = get_neighbors(X, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            cluster_id += 1
            expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples)
    
    return labels


# Load the MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data into a 2D array of shape (n_samples, n_features)
X_train = x_train.reshape((x_train.shape[0], -1))
X_test = x_test.reshape((x_test.shape[0], -1))

# Apply PCA to reduce the dimensions to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Apply DBSCAN to cluster the data
if useLocal == 1:
    X_train_pca = X_train_pca[:100]
    dbscan_labels = dbscan(X_train_pca, eps=10, min_samples=5)
else:
    dbscan = DBSCAN(eps=10, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_train_pca)

# Create a scatter plot of the clustered data
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=dbscan_labels)
plt.title('DBSCAN Clustering of MNIST Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('dbscan_mnist_clusters.png')
plt.show()