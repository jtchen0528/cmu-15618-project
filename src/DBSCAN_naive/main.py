# Apply DBSCAN on PCA-reduced MNIST

# Conda Environment
# conda create -n tf tensorflow
# conda activate tf
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge matplotlib

method = 2
dataset = 'Iris'

from keras.datasets import mnist

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import itertools

class KDTreeNode:
    ''' kdtree for efficient searching on high dimension space.'''
    def __init__(self, point, left_child=None, right_child=None):
        self.point = point
        self.left_child = left_child
        self.right_child = right_child

def construct_kdtree(points, depth=0):
    ''' Build the kdtree on points. Split at "depth" dimension. '''
    if len(points) == 0:
        return None
    
    num_dimensions = len(points[0])
    axis = depth % num_dimensions
    
    sorted_points = sorted(points, key=lambda point: point[axis])
    mid = len(points) // 2
    
    return KDTreeNode(
        point=sorted_points[mid],
        left_child=construct_kdtree(sorted_points[:mid], depth + 1),
        right_child=construct_kdtree(sorted_points[mid+1:], depth + 1)
    )

def search_kdtree(root, target_point, radius):
    ''' Search the kdtree on points. Split at "depth" dimension. '''
    points_within_radius = []
    
    def search_node(node, target_point, depth, radius, points_within_radius):
        if node is None:
            return
        
        node_distance = np.linalg.norm(np.array(node.point) - np.array(target_point))
        if node_distance <= radius:
            points_within_radius.append(node.point)
        
        axis = depth % len(target_point)
        if target_point[axis] - radius <= node.point[axis]:
            search_node(node.left_child, target_point, depth + 1, radius, points_within_radius)
        if target_point[axis] + radius >= node.point[axis]:
            search_node(node.right_child, target_point, depth + 1, radius, points_within_radius)
    
    search_node(root, target_point, 0, radius, points_within_radius)
    return points_within_radius

def label_interpret(Y, skip=[-1]):
    ''' Make labels continuous. '''
    D = {}
    label = 0
    res = []
    for y in Y:
        if y in skip:
            res.append(y)
            continue
        if y not in D:
            D[y] = label
            label += 1
        res.append(D[y])
    return res

def dbscan(X, eps, min_samples):
    ''' Naive DBSCAN. '''
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
    '''
    An implementation of the paper 'On the Hardness and Approximation of Euclidean DBSCAN'.
    '''
    # - Note that any two points in the same cell are at most 
    # - distance 'eps' apart.
    dimension = X.shape[1]
    grid_size = eps / np.sqrt(dimension)

    # - Assign grids to each cells
    grid = {}
    for i, point in enumerate(X):
        grid_coords = tuple((point // grid_size).astype(int))
        if grid_coords not in grid:
            grid[grid_coords] = [i]
        else:
            grid[grid_coords].append(i)

    # - Construct KD Tree
    kdTree = construct_kdtree([k for k,v in grid.items()])
    
    def get_neighbor_coords(coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        ''''''
        dim = len(coords)

        dimOptions = []
        for index in coords:
            dimOptions.append([index-2, index-1, index, index+1, index+2])
            
        def generate_combinations(list_of_lists):
            combos = itertools.product(*list_of_lists)
            tuples = [tuple(combo) for combo in combos]
            return tuples

        neighbors = generate_combinations(dimOptions)
        neighbors.remove(coords)

        return neighbors

    # - Label core cells
    core_cells = set()
    for grid_coords, cell_indices in grid.items():
        
        # In-grid core cell
        if len(cell_indices) >= min_samples+1:
            core_cells.add(grid_coords)
            continue

        # Calculate B(p, eps) 
        neighbors = []
        for neighbor_coords in search_kdtree(kdTree, grid_coords, 2.3):
            if neighbor_coords == grid_coords:
                continue
            if neighbor_coords in grid:
                neighbors += grid[neighbor_coords]
        if len(cell_indices) + len(neighbors) < min_samples+1:
            continue
        
        if min_samples <= 4:
            # Brute-force
            for index in cell_indices:
                numNeighbors = len(cell_indices)
                point = X[index]
                for neighbor_index in neighbors:
                    neighbor = X[neighbor_index]
                    if np.linalg.norm(point - neighbor) <= eps:
                        numNeighbors += 1
                        if numNeighbors >= (min_samples+1):
                            core_cells.add(grid_coords)
                            break
        else:
            # KD Tree approach
            neighbor_kdtree = construct_kdtree(X[neighbors])
            for index in cell_indices:
                point = X[index]
                numNeighbors = len(cell_indices) + len(search_kdtree(neighbor_kdtree, point, eps))
                if numNeighbors >= (min_samples+1):
                    core_cells.add(grid_coords)
                    break

    # - Initial Clusters
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
        # Skip empty cells
        if not cell_indices:
            continue
        
        # Skip non-core cells
        if grid_coords not in core_cells:
            continue 

        # Assign cluster Id to uninitial cells
        if labels[cell_indices[0]] == -1:
            labels[cell_indices] = cluster_id
            cluster_id += 1

        # Expend cell connections based on Bichromatic Closest Pair(BCP)
        
        for neighbor_coords in search_kdtree(kdTree, grid_coords, 2.3):
            if neighbor_coords not in grid:
                continue
            if min_samples <= 4:
                for neighbor_index in grid[neighbor_coords]:
                    neighbor = X[neighbor_index]
                    for index in cell_indices:
                        point = X[index]
                        if np.linalg.norm(point - neighbor) <= eps:
                            if labels[grid[neighbor_coords][0]] == -1:
                                labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                            else:
                                union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                            break
            else:
                neighbor_kdtree = construct_kdtree(X[grid[neighbor_coords]])
                for index in cell_indices:
                    point = X[index]
                    if search_kdtree(neighbor_kdtree, point, eps):
                        if labels[grid[neighbor_coords][0]] == -1:
                            labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                        else:
                            union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                        break


    labels = [find(x) for x in labels]
            
    return labels

if dataset == 'MNIST':
    # Load the MNIST dataset from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data into a 2D array of shape (n_samples, n_features)
    X_train = x_train.reshape((x_train.shape[0], -1))
    X_test = x_test.reshape((x_test.shape[0], -1))
    Y_train = y_train

    # Apply PCA to reduce the dimensions to 2
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    print(f'use MNIST data set, dim = 2(after PCA).')
else:
    iris = load_iris()
    X_train = iris.data
    Y_train = iris.target
    X_train = StandardScaler().fit_transform(X_train)
    print(f'use Iris data set, dim = 4.')

# Apply DBSCAN to cluster the data
if method == 1:
    sub_sample = 10
    X_train = X_train[:sub_sample]
    dbscan_labels = dbscan(X_train, eps=0.1, min_samples=1)
    print(f'naice DBSCAN. {dbscan_labels}')
    print(f'answer. {Y_train[:sub_sample]}')
elif method == 2:
    sub_sample = len(X_train)
    X_train = X_train[:sub_sample]
    # print(f'iris data: {X_train}')
    dbscan_labels = grid_dbscan(X_train, eps=0.5, min_samples=5)
    print(f'grid-based DBSCAN. {label_interpret(dbscan_labels)}')
    print(f'answer. {Y_train[:sub_sample]}')
else:
    dbscan = DBSCAN(eps=10, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_train)
    print(f'sklearn DBSCAN. {dbscan_labels}')

if dataset == 'MNIST':
    # Create a scatter plot of the clustered data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=dbscan_labels)
    plt.title('DBSCAN Clustering of MNIST Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('dbscan_mnist_clusters.png')
    plt.show()
