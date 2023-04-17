import sklearn
from typing import List, Tuple
import numpy as np
from collections import defaultdict
import itertools
from sklearn import metrics
import os
import time
import json

import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Lock

import pickle

globalLock = defaultdict(Lock)

class KDTreeNode:
    ''' kdtree for efficient searching on high dimension space. '''
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
    # - we dont need sort here, just quick select and split based on pivot.
    sorted_points = sorted(points, key=lambda point: point[axis])
    mid = len(points) // 2
    node = KDTreeNode(point=sorted_points[mid])
    node.left_child = construct_kdtree(sorted_points[:mid], depth + 1)
    node.right_child = construct_kdtree(sorted_points[mid+1:], depth + 1)
    return node


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


def align_labels(Y, _Y, skip=[-1]):
    ''' Align labels. '''
    if len(Y) != len(_Y):
        print(f'EXCEPTION: prediction(len={len(Y)}) and ground truth(len={len(Y)}) has different length.')
        return Y
    D = {}
    res = []
    for i in range(len(Y)):
        y = Y[i]
        if y not in skip:
            res.append(y)
            continue
        if y not in D:
            D[y] = _Y[i]
        res.append(D[y])
    return res

def get_grid_coords(args):
    i, point, grid_size = args
    grid_coords = tuple((point // grid_size).astype(int))
    return tuple((i, grid_coords))

def parallel_construct_grid(X, grid_size, num_processes=4):
    # init
    grid = defaultdict(list)
    # create pool: force the #thread to be 1 ba there's no paralell benefits
    pool = mp.Pool(num_processes)
    # create args
    args = [(i, point, grid_size) for i, point in enumerate(X)]
    # passing function and args
    ret = pool.map(get_grid_coords, args)
    # assign results
    for i, grid_coords in ret:
        grid[grid_coords].append(i)
        
    return grid

def is_in_grid_core_cell(args):
    grid_id, cell_indices, min_samples = args
    return tuple((grid_id, len(cell_indices) > min_samples+1))

def parallel_find_in_grid_core_cell(grid, grid_id, min_samples, num_processes=4):
    # init
    core_cells = np.zeros(len(grid), dtype=bool)
    # create pool
    pool = mp.Pool(num_processes)
    # create args
    args = [(grid_id[grid_coords], cell_indices, min_samples) for grid_coords, cell_indices in grid.items()]
    # passing function and args
    ret = pool.map(is_in_grid_core_cell, args)
    # process the results
    indices, truths = zip(*ret)
    core_cells[list(indices)] = truths

    return core_cells


def is_out_grid_core_cell(args):
    grid_coords, cell_indices, neighbor_coords_list, grid, min_samples, X, grid_id, eps = args
    # get neighbors
    neighbors = []
    for neighbor_coords in neighbor_coords_list:
        if neighbor_coords == grid_coords:
            continue
        neighbors += grid[neighbor_coords]
    
    # filter those mavericks
    if len(cell_indices) + len(neighbors) < min_samples+1:
        return tuple((grid_id[grid_coords], False))
    
    # Check number of connections
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
                        return tuple((grid_id[grid_coords], True))
                        done = True
    else:
        # KD Tree approach
        neighbor_kdtree = construct_kdtree(X[neighbors])
        for index in cell_indices:
            point = X[index]
            numNeighbors = len(cell_indices) + len(search_kdtree(neighbor_kdtree, point, eps))
            if numNeighbors >= (min_samples+1):
                return tuple((grid_id[grid_coords], True))
    
    return tuple((grid_id[grid_coords], False))
    

def parallel_find_out_grid_core_cell(
        kdTree,
        eps_coverage,
        grid,
        min_samples,
        X,
        grid_id,
        eps,
        core_cells, 
        num_processes
    ):
    # create pool
    pool = mp.Pool(num_processes)
    # create args
    args = [(grid_coords, cell_indices, search_kdtree(kdTree, grid_coords, eps_coverage), grid, min_samples, X, grid_id, eps)
            for i, (grid_coords, cell_indices) in enumerate(grid.items()) if core_cells[i] == False]
    # passing function and args
    ret = pool.map(is_out_grid_core_cell, args)
    # process the results
    indices, truths = zip(*ret)
    core_cells[list(indices)] = truths
        
    return core_cells



def is_clustering(args):
    cluster_id, grid_coords, cell_indices, neighbor_coords_list, grid, min_samples, X, eps = args
    res = []
    # Expend cell connections based on Bichromatic Closest Pair(BCP)
    for neighbor_coords in neighbor_coords_list:
        if min_samples <= 4:
            done = False
            for neighbor_index in grid[neighbor_coords]:
                if done:
                    break
                neighbor = X[neighbor_index]
                for index in cell_indices:
                    point = X[index]
                    if np.linalg.norm(point - neighbor) <= eps:
                        res.append((grid_coords, neighbor_coords, cluster_id))
                        done = True
                        break
        else:
            neighbor_kdtree = construct_kdtree(X[grid[neighbor_coords]])
            for index in cell_indices:
                point = X[index]
                if search_kdtree(neighbor_kdtree, point, eps):
                    res.append((grid_coords, neighbor_coords, cluster_id))
                    break
    return res
        

def parallel_clustering(
        kdTree,
        eps_coverage,
        grid,
        min_samples,
        X,
        grid_id,
        eps,
        core_cells, 
        num_processes
    ):
    # Init
    labels = np.full(X.shape[0], -1)
    for cluster_id, (grid_coords, cell_indices) in enumerate(grid.items()):
        if core_cells[cluster_id] == True:
            labels[cell_indices] = cluster_id
    
    # Define Union and Find
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
    
    # create pool
    pool = mp.Pool(num_processes)
    # create args
    args = [(cluster_id, grid_coords, cell_indices, search_kdtree(kdTree, grid_coords, eps_coverage),
            grid,
            min_samples,
            X,
            eps)
            for cluster_id, (grid_coords, cell_indices) in enumerate(grid.items()) if core_cells[cluster_id] == True]
    # passing function and args
    ret = pool.map(is_clustering, args)
    # process the results
    for res in ret:
        for (grid_coords, neighbor_coords, cluster_id) in res:
            if labels[grid[neighbor_coords][0]] == -1:
                labels[grid[neighbor_coords]] = labels[grid[grid_coords][0]]
            else:
                if grid_id[grid_coords] > grid_id[neighbor_coords]:
                    union(grid[grid_coords][0], grid[neighbor_coords][0])
        
    labels = [find(x) for x in labels]

    return labels


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, model='grid', n_threads = 4, omp = False):
        self.eps = eps
        self.min_samples = min_samples
        self.model = model
        self.labels = None
        self.time_log = {}
        self.n_threads = n_threads
        print(f'Initialize DBSCAN-{self.model} with eps:{self.eps} min_samples:{self.min_samples}')
    
    def log_time(self, entry_name, start_time):
        print(f'Logging {entry_name}')
        self.time_log[entry_name] = time.time() - start_time
    
    def fit(self, X):
        if self.model == 'grid':
            self.labels = self._fit_grid(X)
        if self.model == 'grid_mp':
            self.labels = self._fit_grid_mp(X)
        if self.model == 'sklearn':
            _dbscan = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.labels = _dbscan.fit_predict(X)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def evaluate(self, X):
        return self.fit_predict(X)
    
    def _fit_grid(self, X):
        dimension = X.shape[1]
        grid_size = self.eps / np.sqrt(dimension)
        eps_coverage = np.sqrt(dimension)+1

        # - Assign grids to each cells
        start_time = time.time()
        
        grid = defaultdict(list)
        for i, point in enumerate(X):
            grid_coords = tuple((point // grid_size).astype(int))
            grid[grid_coords].append(i)
        
        grid_id = {k:i for i, (k, _) in enumerate(list(grid.items()))}
        numGrids = len(grid_id)
        self.log_time("dbsacn_grid-init grid", start_time)
        
        start_time = time.time()
        kdTree = construct_kdtree([k for k,v in grid.items()])
        self.log_time("dbsacn_grid-construct kdtree", start_time)

        # - Label core cells
        start_time = time.time()
        core_cells = [False] * numGrids
        for grid_coords, cell_indices in grid.items():
            # In-grid core cell
            if len(cell_indices) >= self.min_samples+1:
                core_cells[grid_id[grid_coords]] = True
                continue

            # Get neighbors
            neighbors = []
            for neighbor_coords in search_kdtree(kdTree, grid_coords, eps_coverage):
                if neighbor_coords == grid_coords:
                    continue
                neighbors += grid[neighbor_coords]
            if len(cell_indices) + len(neighbors) < self.min_samples+1:
                continue
            
            # Check number of connections
            if self.min_samples <= 4:
                # Brute-force
                for index in cell_indices:
                    numNeighbors = len(cell_indices)
                    point = X[index]
                    for neighbor_index in neighbors:
                        neighbor = X[neighbor_index]
                        if np.linalg.norm(point - neighbor) <= self.eps:
                            numNeighbors += 1
                            if numNeighbors >= (self.min_samples+1):
                                core_cells[grid_id[grid_coords]] = True
                                break
            else:
                # KD Tree approach
                neighbor_kdtree = construct_kdtree(X[neighbors])
                for index in cell_indices:
                    point = X[index]
                    numNeighbors = len(cell_indices) + len(search_kdtree(neighbor_kdtree, point, self.eps))
                    if numNeighbors >= (self.min_samples+1):
                        core_cells[grid_id[grid_coords]] = True
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
        for grid_coords, cell_indices in grid.items():
            # Skip non-core cells
            if core_cells[grid_id[grid_coords]] == False:
                continue 

            # Assign cluster Id
            if labels[cell_indices[0]] == -1:
                labels[cell_indices] = cluster_id
                cluster_id += 1

            # Expend cell connections based on Bichromatic Closest Pair(BCP)
            for neighbor_coords in search_kdtree(kdTree, grid_coords, eps_coverage):
                if self.min_samples <= 4:
                    for neighbor_index in grid[neighbor_coords]:
                        neighbor = X[neighbor_index]
                        for index in cell_indices:
                            point = X[index]
                            if np.linalg.norm(point - neighbor) <= self.eps:
                                if labels[grid[neighbor_coords][0]] == -1:
                                    labels[grid[neighbor_coords]] = labels[cell_indices[0]]
                                else:
                                    if grid_id[grid_coords] > grid_id[neighbor_coords]:
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
                                if grid_id[grid_coords] > grid_id[neighbor_coords]:
                                    union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                            break
        self.log_time("dbscan_grid-clustering", start_time)
        
        start_time = time.time()
        labels = [find(x) for x in labels]
        self.log_time("dbscan_grid-find labels", start_time)
                
        return labels
    
    def _fit_grid_mp(self, X):
        dimension = X.shape[1]
        grid_size = self.eps / np.sqrt(dimension)
        eps_coverage = np.sqrt(dimension)+1
        
        # - Assign grids to each cells
        
        numSamples = len(X)
        
        start_time = time.time()
        grid = parallel_construct_grid(X, grid_size, self.n_threads)
        self.log_time("dbsacn_grid-init grid", start_time)
        
        start_time = time.time()
        grid_id = {}
        for i, (k, _) in enumerate(list(grid.items())): 
            grid_id[k] = i
        self.log_time("dbsacn_grid-Calculate grid grid_id", start_time)
        
        numGrids = len(grid_id)
        
        start_time = time.time()
        kdTree = construct_kdtree([k for k,v in grid.items()])
        self.log_time("dbsacn_grid-construct kdtree", start_time)

        # - Label core cells
        start_time = time.time()
        core_cells = parallel_find_in_grid_core_cell(grid, grid_id, self.min_samples, self.n_threads)
        # print(core_cells)
        self.log_time("dbscan_grid-label in-grid core cells", start_time)
        
        start_time = time.time()
        core_cells = parallel_find_out_grid_core_cell(
            kdTree,
            eps_coverage,
            grid,
            self.min_samples,
            X,
            grid_id,
            self.eps,
            core_cells,
            self.n_threads
        )
        self.log_time("dbscan_grid-label out-grid core cells", start_time)
        
        # - Clustering
        start_time = time.time()
        labels = parallel_clustering(
            kdTree,
            eps_coverage,
            grid,
            self.min_samples,
            X,
            grid_id,
            self.eps,
            core_cells,
            self.n_threads
        )
        self.log_time("dbscan_grid-clustering", start_time)
                
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