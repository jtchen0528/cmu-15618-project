# Apply DBSCAN on PCA-reduced MNIST

# Conda Environment
# conda create -n tf tensorflow
# conda activate tf
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge matplotlib

method = 2

from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from typing import List, Tuple
import numpy as np
from collections import defaultdict


def dbscan(X, eps, min_samples):
    labels = np.zeros(len(X))
    cluster_id = 0
    
    def get_neighbors(X, i, eps):
        
        def euclidean_distance(x1, x2):
            return np.sqrt(np.sum((x1 - x2)**2))

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


def grid_dbscan(X: np.ndarray, eps: float, min_samples: int) -> List[int]:
    """
    Performs grid-based DBSCAN clustering on the input data.
    :param X: The input data as a numpy array.
    :param eps: The radius of the neighborhood to be considered.
    :param min_samples: The minimum number of samples required for a cluster.
    :return: A list of cluster labels for each data point.
    """
    grid_size = eps / np.sqrt(2)
    
    grid = {}
    for i, point in enumerate(X):
        grid_coords = tuple((point // grid_size).astype(int))
        if grid_coords not in grid:
            grid[grid_coords] = [i]
        else:
            grid[grid_coords].append(i)

    def get_neighbor_coords(coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = coords
        return [(x-2, y-1), (x-2, y), (x-2, y+1),\
                (x-1, y-2), (x-1, y-1), (x-1, y), (x-1, y+1), (x-1, y+2),\
                (x, y-2), (x, y-1), (x, y), (x, y+1), (x, y+2),\
                (x+1, y-2), (x+1, y-1), (x+1, y), (x+1, y+1), (x+1, y+2),\
                (x+2, y-1), (x+2, y), (x+2, y+1)]

    # Find core grids
    core_grids = set()
    for grid_coords, cell_indices in grid.items():
        if len(cell_indices) >= min_samples+1:
            core_grids.add(grid_coords)
            continue
        
        neighbors = []
        for neighbor_coords in get_neighbor_coords(grid_coords):
            if neighbor_coords in grid:
                neighbors += grid[neighbor_coords]
        
        if len(cell_indices) + len(neighbors) < min_samples+1:
            continue
        
        # TODO: This can be optimized
        for index in cell_indices:
            numNeighbors = len(cell_indices)
            point = X[index]
            for neighbor_index in neighbors:
                neighbor = X[neighbor_index]
                if np.linalg.norm(point - neighbor) <= eps:
                    numNeighbors += 1
                    if numNeighbors >= min_samples+1:
                        core_grids.add(grid_coords)
                        break
    
    # Expand the cells
    labels = np.full(X.shape[0], -1)
    
    D = {}
    def find(x):
        while x in D and x != D[x]:
            x = D[x]
        if x not in D:
            D[x] = x
        return x
    def union(a, b):
        a, b = find(a), find(b)
        D[a] = min(a, b)
        D[b] = min(a, b)
        
    cluster_id = 0
    for grid_coords, cell_indices in grid.items():
        # Skip non-core cells
        if grid_coords not in core_grids:
            continue 
        
        if not cell_indices:
            continue
        
        # Assign cluster Id to uninitial cells
        if labels[cell_indices[0]] == -1:
            labels[cell_indices] = cluster_id
            cluster_id += 1
        
        # Expend cell connections
        for neighbor_coords in get_neighbor_coords(grid_coords):
            if neighbor_coords not in grid:
                continue
            min_distance = 10**9
            for neighbor_index in grid[neighbor_coords]:
                neighbor = X[neighbor_index]
                for index in cell_indices:
                    point = X[index]
                    min_distance = min(min_distance, np.linalg.norm(point - neighbor))
            if min_distance <= eps:
                if not grid[neighbor_coords]:
                    continue
                if labels[grid[neighbor_coords][0]] == -1:
                    labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                else:
                    union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
    
    labels = [find(x) for x in labels]
            
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
if method == 1:
    X_train_pca = X_train_pca[:10]
    dbscan_labels = dbscan(X_train_pca, eps=200, min_samples=1)
elif method == 2:
    X_train_pca = X_train_pca[:10]
    dbscan_labels = grid_dbscan(X_train_pca, eps=200, min_samples=1)
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