import sklearn
from typing import List, Tuple
import numpy as np
from collections import defaultdict
import itertools
from sklearn import metrics
import os
import time
import json

from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_set_num_threads, omp_get_thread_num, omp_get_num_threads, omp_get_wtime

class KDTreeNode:
    ''' kdtree for efficient searching on high dimension space.'''
    def __init__(self, point, left_child=None, right_child=None):
        self.point = point
        self.left_child = left_child
        self.right_child = right_child

#@omp @njit
def construct_kdtree(points, depth=0):
    ''' Build the kdtree on points. Split at "depth" dimension. '''
    if len(points) == 0:
        return None
    num_dimensions = len(points[0])
    axis = depth % num_dimensions
    sorted_points = sorted(points, key=lambda point: point[axis])
    mid = len(points) // 2
    node = KDTreeNode(point=sorted_points[mid])
    #@omp with openmp("single"):
        #@omp with openmp("task"):
    node.left_child = construct_kdtree(sorted_points[:mid], depth + 1)
        #@omp with openmp("task"):
    node.right_child = construct_kdtree(sorted_points[mid+1:], depth + 1)
    return node

#@omp @njit
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
        #@omp with openmp("single"):
            #@omp with openmp("task"):
        if target_point[axis] - radius <= node.point[axis]: 
            search_node(node.left_child, target_point, depth + 1, radius, points_within_radius)
            #@omp with openmp("task"):
        if target_point[axis] + radius >= node.point[axis]:
            search_node(node.right_child, target_point, depth + 1, radius, points_within_radius)

    #@omp with openmp("single"):
        #@omp with openmp("task"):
    search_node(root, target_point, 0, radius, points_within_radius)
    return points_within_radius

def align_labels(Y, _Y, skip=[-1]):
    ''' Align labels. '''
    if len(Y) != len(_Y):
        print(f'EXCEPTION: prediction(len={len(Y)}) and ground truth(len={len(Y)}) has different length.')
        return Y
    D = {}
    res = []
    for i in range(len(Y)):
        y = Y[i]
        if y in skip:
            res.append(y)
            continue
        if y not in D:
            D[y] = _Y[i]
        res.append(D[y])
    return res

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, model='naive'):
        self.eps = eps
        self.min_samples = min_samples
        self.model = model
        self.labels = None
        self.time_log = {}
        print(f'Initialize DBSCAN-{self.model} with eps:{self.eps} min_samples:{self.min_samples}')
    
    def log_time(self, entry_name, start_time):
        self.time_log[entry_name] = time.time() - start_time
    
    def fit(self, X):
        if self.model == 'naive':
            self.labels = self._fit_naive(X)
        if self.model == 'grid':
            self.labels = self._fit_grid(X)
        if self.model == 'grid_omp':
            self.labels = self._fit_grid_omp(X)
        if self.model == 'sklearn':
            _dbscan = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.labels = _dbscan.fit_predict(X)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def evaluate(self, X):
        return self.fit_predict(X)
        
    def _fit_naive(self, X):
        '''
        @WARN: naive implementation is O(N^2). I never saw its execution completed.
        '''
        labels = np.full(X.shape[0], 0)
        cluster_id = 0

        def get_neighbors(X, i, eps):
            neighbors = []
            for j in range(len(X)):
                if i != j and np.linalg.norm(X[i] - X[j]) <= eps:
                    neighbors.append(j)
            return neighbors

        def expand_cluster(X, i, neighbors, cluster_id, eps, min_samples):
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
            
        start_time = time.time()
        for i in range(len(X)):
            if labels[i] != 0:
                continue
            neighbors = get_neighbors(X, i, self.eps)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                cluster_id += 1
                expand_cluster(X, i, neighbors, cluster_id, self.eps, self.min_samples)
        self.log_time("dbscan_naive-clustering", start_time)
        
        return labels
    
    def _fit_grid(self, X):
        dimension = X.shape[1]
        grid_size = self.eps / np.sqrt(dimension)

        # - Assign grids to each cells
        start_time = time.time()
        grid = defaultdict(list)
        numSamples = len(X)
        ID = {}
        grid_index = 0
        #@omp with openmp("for"):
        for i, point in enumerate(X):
            grid_coords = tuple((point // grid_size).astype(int))
            #@omp with openmp("critical"):
            if grid_coords not in grid:
                ID[grid_coords] = grid_index
                grid_index += 1
            grid[grid_coords].append(i)
        self.log_time("dbsacn_grid-init grid", start_time)

        numGrids = grid_index
        
        start_time = time.time()
        kdTree = construct_kdtree([k for k,v in grid.items()])
        self.log_time("dbsacn_grid-construct kdtree", start_time)

        # - Label core cells
        start_time = time.time()
        core_cells = [False] * numGrids
        #@omp with openmp("for"):
        for grid_coords, cell_indices in grid.items():
            # In-grid core cell
            if len(cell_indices) >= self.min_samples+1:
                core_cells[ID[grid_coords]] = True
                continue

            # Get neighbors
            neighbors = []
            # omp for, critical: neighbors
            #@omp with openmp("for"):
            for neighbor_coords in search_kdtree(kdTree, grid_coords, 2.3):
                if neighbor_coords == grid_coords:
                    continue
                if neighbor_coords in grid:
                    #@omp with openmp("critical"):
                    neighbors += grid[neighbor_coords]
            if len(cell_indices) + len(neighbors) < self.min_samples+1:
                continue
            
            # Check number of connections
            if self.min_samples <= 4:
                # Brute-force
                #@omp with openmp("for"):
                for index in cell_indices:
                    numNeighbors = len(cell_indices)
                    point = X[index]
                    #@omp with openmp("for"):
                    for neighbor_index in neighbors:
                        neighbor = X[neighbor_index]
                        if np.linalg.norm(point - neighbor) <= self.eps:
                            numNeighbors += 1
                            if numNeighbors >= (self.min_samples+1):
                                core_cells[ID[grid_coords]] = True
                                break
            else:
                # KD Tree approach
                neighbor_kdtree = construct_kdtree(X[neighbors])
                #@omp with openmp("for"):
                for index in cell_indices:
                    point = X[index]
                    numNeighbors = len(cell_indices) + len(search_kdtree(neighbor_kdtree, point, self.eps))
                    if numNeighbors >= (self.min_samples+1):
                        core_cells[ID[grid_coords]] = True
                        break
        self.log_time("dbscan_grid-label core cells", start_time)
        
        # - Clustering
        start_time = time.time()
        labels = np.full(X.shape[0], -1)
        
        # Union and Find for clustering
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
        #@omp with openmp("for"):
        for grid_coords, cell_indices in grid.items():
            # Skip empty cells
            if not cell_indices:
                continue
            
            # Skip non-core cells
            if core_cells[ID[grid_coords]] == False:
                continue 

            # Assign cluster Id
            if labels[cell_indices[0]] == -1:
                labels[cell_indices] = cluster_id
                cluster_id += 1

            # Expend cell connections based on Bichromatic Closest Pair(BCP)
            #@omp with openmp("for"):
            for neighbor_coords in search_kdtree(kdTree, grid_coords, 2.3):
                if neighbor_coords not in grid:
                    continue
                if self.min_samples <= 4:
                    for neighbor_index in grid[neighbor_coords]:
                        neighbor = X[neighbor_index]
                        for index in cell_indices:
                            point = X[index]
                            if np.linalg.norm(point - neighbor) <= self.eps:
                                if labels[grid[neighbor_coords][0]] == -1:
                                    labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                                else:
                                    if ID[grid_coords] > ID[neighbor_coords]:
                                        #@omp with openmp("critical"):
                                        union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                                break
                else:
                    neighbor_kdtree = construct_kdtree(X[grid[neighbor_coords]])
                    for index in cell_indices:
                        point = X[index]
                        if search_kdtree(neighbor_kdtree, point, self.eps):
                            if labels[grid[neighbor_coords][0]] == -1:
                                labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                            else:
                                if ID[grid_coords] > ID[neighbor_coords]:
                                    #@omp with openmp("critical"):
                                    union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                            break
        self.log_time("dbscan_grid-clustering", start_time)
        
        start_time = time.time()
        labels = [find(x) for x in labels]
        self.log_time("dbscan_grid-find labels", start_time)
                
        return labels
    
    def _fit_grid_omp(self, X):
        dimension = X.shape[1]
        grid_size = self.eps / np.sqrt(dimension)

        # - Assign grids to each cells
        start_time = time.time()
        grid = defaultdict(list)
        numSamples = len(X)
        ID = {}
        grid_index = 0
        with openmp("parallel for"):
            for i, point in enumerate(X):
                grid_coords = tuple((point // grid_size).astype(int))
                with openmp("critical"):
                    if grid_coords not in grid:
                        ID[grid_coords] = grid_index
                        grid_index += 1
                    grid[grid_coords].append(i)
        self.log_time("dbsacn_grid-init grid", start_time)

        numGrids = grid_index
        
        start_time = time.time()
        kdTree = construct_kdtree([k for k,v in grid.items()])
        self.log_time("dbsacn_grid-construct kdtree", start_time)

        # - Label core cells
        start_time = time.time()
        core_cells = [False] * numGrids
        with openmp("parallel for"):
            for grid_coords, cell_indices in grid.items():
                # In-grid core cell
                if len(cell_indices) >= self.min_samples+1:
                    core_cells[ID[grid_coords]] = True
                    continue

                # Get neighbors
                neighbors = []
                # omp for, critical: neighbors
                with openmp("for"):
                    for neighbor_coords in search_kdtree(kdTree, grid_coords, 2.3):
                        if neighbor_coords == grid_coords:
                            continue
                        if neighbor_coords in grid:
                            with openmp("critical"):
                                neighbors += grid[neighbor_coords]
                    if len(cell_indices) + len(neighbors) < self.min_samples+1:
                        continue
                
                # Check number of connections
                if self.min_samples <= 4:
                    # Brute-force
                    with openmp("for"):
                        for index in cell_indices:
                            numNeighbors = len(cell_indices)
                            point = X[index]
                            with openmp("for"):
                                for neighbor_index in neighbors:
                                    neighbor = X[neighbor_index]
                                    if np.linalg.norm(point - neighbor) <= self.eps:
                                        numNeighbors += 1
                                        if numNeighbors >= (self.min_samples+1):
                                            core_cells[ID[grid_coords]] = True
                                            break
                else:
                    # KD Tree approach
                    neighbor_kdtree = construct_kdtree(X[neighbors])
                    with openmp("for"):
                        for index in cell_indices:
                            point = X[index]
                            numNeighbors = len(cell_indices) + len(search_kdtree(neighbor_kdtree, point, self.eps))
                            if numNeighbors >= (self.min_samples+1):
                                core_cells[ID[grid_coords]] = True
                                break
        self.log_time("dbscan_grid-label core cells", start_time)
        
        # - Clustering
        start_time = time.time()
        labels = np.full(X.shape[0], -1)
        
        # Union and Find for clustering
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
        with openmp("parallel for"):
            for grid_coords, cell_indices in grid.items():
                # Skip empty cells
                if not cell_indices:
                    continue
                
                # Skip non-core cells
                if core_cells[ID[grid_coords]] == False:
                    continue 

                # Assign cluster Id
                if labels[cell_indices[0]] == -1:
                    labels[cell_indices] = cluster_id
                    cluster_id += 1

                # Expend cell connections based on Bichromatic Closest Pair(BCP)
                with openmp("for"):
                    for neighbor_coords in search_kdtree(kdTree, grid_coords, 2.3):
                        if neighbor_coords not in grid:
                            continue
                        if self.min_samples <= 4:
                            for neighbor_index in grid[neighbor_coords]:
                                neighbor = X[neighbor_index]
                                for index in cell_indices:
                                    point = X[index]
                                    if np.linalg.norm(point - neighbor) <= self.eps:
                                        if labels[grid[neighbor_coords][0]] == -1:
                                            labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                                        else:
                                            if ID[grid_coords] > ID[neighbor_coords]:
                                                with openmp("critical"):
                                                    union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                                        break
                        else:
                            neighbor_kdtree = construct_kdtree(X[grid[neighbor_coords]])
                            for index in cell_indices:
                                point = X[index]
                                if search_kdtree(neighbor_kdtree, point, self.eps):
                                    if labels[grid[neighbor_coords][0]] == -1:
                                        labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                                    else:
                                        if ID[grid_coords] > ID[neighbor_coords]:
                                            with openmp("critical"):
                                                union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                                    break
        self.log_time("dbscan_grid-clustering", start_time)
        
        start_time = time.time()
        labels = [find(x) for x in labels]
        self.log_time("dbscan_grid-find labels", start_time)
                
        return labels
    
    def show_metrics(self, X, Y, output_dir):
        # View results
        print("[RESULT]")
        predicted_Y_train = self.evaluate(X)
        predicted_Y_train = align_labels(predicted_Y_train, Y)
        classification = predicted_Y_train
        print('Homogeneity: {}'.format(metrics.homogeneity_score(Y, classification)))
        print('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y_train)))

        with open(os.path.join(output_dir, "accuracy.txt"), "w") as f:
            f.write('Homogeneity: {}\n'.format(metrics.homogeneity_score(Y, classification)))
            f.write('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y_train)))
    
    def write_time_log(self, output_dir):
        with open(os.path.join(output_dir, "time_log.txt"), "w") as f:
            f.write(json.dumps(self.time_log, indent = 4))