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
from multiprocessing import Process, Queue


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
        if y in skip:
            res.append(y)
            continue
        if y not in D:
            D[y] = _Y[i]
        res.append(D[y])
    return res


def construct_grid(task_queue, done_queue, grid_size, X):
    while True:
        # get a task from the task queue
        i = task_queue.get()
        if i is None:
            # if the task is None, it means we are done
            break

        # process the point
        point = X[i]
        grid_coords = tuple((point // grid_size).astype(int))
        done_queue.put((grid_coords, i))


def parallel_construct_grid(X, grid_size, num_processes=4):
    # init
    grid = defaultdict(list)

    # create a task queue and a done queue
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # add tasks to the task queue
    for i in range(len(X)):
        task_queue.put(i)

    # add a stop signal to the task queue for each process
    for _ in range(num_processes):
        task_queue.put(None)

    # create and start the worker processes
    processes = [mp.Process(target=construct_grid, args=(task_queue, done_queue, grid_size, X))
                 for _ in range(num_processes)]
    for process in processes:
        process.start()

    # process the results
    for _ in range(len(X)):
        grid_coords, i = done_queue.get()
        grid[grid_coords].append(i)

    # wait for the worker processes to finish
    for process in processes:
        process.join()

    return grid


def find_in_grid_core_cell(task_queue, done_queue, ID, min_samples):
    while True:
        # get a task from the task queue
        task = task_queue.get()
        if task is None:
            # if the task is None, it means we are done
            break

        # process the task
        (grid_coords, cell_indices) = task
        done_queue.put((ID[grid_coords], len(cell_indices) > min_samples+1))


def parallel_find_in_grid_core_cell(grid, ID, min_samples, num_processes):
    # init
    core_cells = np.zeros(len(grid), dtype=bool)
    
    # create a task queue and a done queue
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    # add tasks to the task queue
    for grid_coords, cell_indices in grid.items():
        task_queue.put((grid_coords, cell_indices))

    # add a stop signal to the task queue for each process
    for _ in range(num_processes):
        task_queue.put(None)
        
    # create and start the worker processes
    processes = [mp.Process(target=find_in_grid_core_cell, args=(task_queue, done_queue, ID, min_samples))
                 for _ in range(num_processes)]
    for process in processes:
        process.start()
        
    # process the results
    for _ in range(len(core_cells)):
        index, truth = done_queue.get()
        core_cells[index] = truth

    # wait for the worker processes to finish
    for process in processes:
        process.join()

    return core_cells


def find_out_grid_core_cell(
    task_queue,
    done_queue,
    kdTree,
    eps_coverage,
    grid,
    min_samples,
    X,
    ID,
    eps):
    while True:
        # get a task from the task queue
        task = task_queue.get()
        if task is None:
            # if the task is None, it means we are done
            break

        # process the task
        (grid_coords, cell_indices) = task
        done = False
        
        # get neighbors
        neighbors = []
        for neighbor_coords in search_kdtree(kdTree, grid_coords, eps_coverage):
            if neighbor_coords == grid_coords:
                continue
            neighbors += grid[neighbor_coords]
        
        # filter those mavericks
        if len(cell_indices) + len(neighbors) < min_samples+1:
            done_queue.put((ID[grid_coords], False))
            done = True
            continue
        
        # Check number of connections
        if min_samples <= 4:
            # Brute-force
            for index in cell_indices:
                if done:
                    break
                numNeighbors = len(cell_indices)
                point = X[index]
                for neighbor_index in neighbors:
                    if done:
                        break
                    neighbor = X[neighbor_index]
                    if np.linalg.norm(point - neighbor) <= eps:
                        numNeighbors += 1
                        if numNeighbors >= (min_samples+1):
                            done_queue.put((ID[grid_coords], True))
                            done = True
        else:
            # KD Tree approach
            neighbor_kdtree = construct_kdtree(X[neighbors])
            for index in cell_indices:
                if done:
                    break
                point = X[index]
                numNeighbors = len(cell_indices) + len(search_kdtree(neighbor_kdtree, point, eps))
                if numNeighbors >= (min_samples+1):
                    done_queue.put((ID[grid_coords], True))
                    done = True
        
        done_queue.put((ID[grid_coords], len(cell_indices) > min_samples+1))


def parallel_find_out_grid_core_cell(
        kdTree,
        eps_coverage,
        grid,
        min_samples,
        X,
        ID,
        eps,
        core_cells, 
        num_processes
    ):
    
    # create a task queue and a done queue
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    # add tasks to the task queue
    send_count = 0
    for i, (grid_coords, cell_indices) in enumerate(grid.items()):
        if core_cells[i] == False:
            send_count += 1
            task_queue.put((grid_coords, cell_indices))

    # add a stop signal to the task queue for each process
    for _ in range(num_processes):
        task_queue.put(None)
        
    # create and start the worker processes
    processes = [mp.Process(target=find_out_grid_core_cell, args=(
                 task_queue,
                 done_queue,
                 kdTree,
                 eps_coverage,
                 grid,
                 min_samples,
                 X,
                 ID,
                 eps))
                 for _ in range(num_processes)]
    for process in processes:
        process.start()
        
    # process the results
    for _ in range(send_count):
        index, truth = done_queue.get()
        core_cells[index] = truth

    # wait for the worker processes to finish
    for process in processes:
        process.join()

    return core_cells
         
   
def clustering(    
    task_queue,
    done_queue,
    kdTree,
    eps_coverage,
    grid,
    min_samples,
    X,
    ID,
    eps):
    while True:
        # get a task from the task queue
        task = task_queue.get()
        if task is None:
            # if the task is None, it means we are done
            break

        # process the task
        (cluster_id, grid_coords, cell_indices) = task
        res = []
        
        # Expend cell connections based on Bichromatic Closest Pair(BCP)
        for neighbor_coords in search_kdtree(kdTree, grid_coords, eps_coverage):
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
        done_queue.put(res)
            

def parallel_clustering(
        kdTree,
        eps_coverage,
        grid,
        min_samples,
        X,
        ID,
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

    # create a task queue and a done queue
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    # add tasks to the task queue
    send_count = 0
    for cluster_id, (grid_coords, cell_indices) in enumerate(grid.items()):
        if core_cells[cluster_id] == True and len(cell_indices) > 0:
            send_count += 1
            task_queue.put((cluster_id, grid_coords, cell_indices))

    # add a stop signal to the task queue for each process
    for _ in range(num_processes):
        task_queue.put(None)
        
    # create and start the worker processes
    processes = [mp.Process(target=clustering, args=(
                 task_queue,
                 done_queue,
                 kdTree,
                 eps_coverage,
                 grid,
                 min_samples,
                 X,
                 ID,
                 eps))
                 for _ in range(num_processes)]
    for process in processes:
        process.start()
        
    # process the results
    for _ in range(send_count):
        res = done_queue.get()
        for grid_coords, neighbor_coords, _ in res:
            if labels[grid[neighbor_coords][0]] == -1:
                labels[grid[neighbor_coords]] = labels[grid[grid_coords][0]]
            else:
                union(grid[grid_coords][0], grid[neighbor_coords][0])
        
    labels = [find(x) for x in labels]
    
    # wait for the worker processes to finish
    for process in processes:
        process.join()

    return labels


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, model='grid', n_threads = 4, omp = False):
        self.eps = eps
        self.min_samples = min_samples
        self.model = model
        self.labels = None
        self.time_log = {}
        self.n_threads = n_threads
        self.omp = omp
        print(f'Initialize DBSCAN-{self.model} with eps:{self.eps} min_samples:{self.min_samples}')
    
    def log_time(self, entry_name, start_time):
        print(f'Logging {entry_name}')
        self.time_log[entry_name] = time.time() - start_time
    
    def fit(self, X):
        if self.model == 'grid':
            self.labels = self._fit_grid(X)
        if self.model == 'grid_parallel':
            self.labels = self._fit_grid_parallel(X)
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
        
        ID = {k:i for i, (k, _) in enumerate(list(grid.items()))}
        numGrids = len(ID)
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
                core_cells[ID[grid_coords]] = True
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
                                core_cells[ID[grid_coords]] = True
                                break
            else:
                # KD Tree approach
                neighbor_kdtree = construct_kdtree(X[neighbors])
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
        for grid_coords, cell_indices in grid.items():
            # Skip non-core cells
            if core_cells[ID[grid_coords]] == False:
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
                                    if ID[grid_coords] > ID[neighbor_coords]:
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
                                    union(labels[cell_indices[0]], labels[grid[neighbor_coords][0]])
                            break
        self.log_time("dbscan_grid-clustering", start_time)
        
        start_time = time.time()
        labels = [find(x) for x in labels]
        self.log_time("dbscan_grid-find labels", start_time)
                
        return labels
    
    def _fit_grid_parallel(self, X):
        dimension = X.shape[1]
        grid_size = self.eps / np.sqrt(dimension)
        eps_coverage = np.sqrt(dimension)+1
        
        # - Assign grids to each cells
        start_time = time.time()
        numSamples = len(X)
        
        grid = parallel_construct_grid(X, grid_size, self.n_threads)
        
        ID = {}
        for i, (k, _) in enumerate(list(grid.items())): 
            ID[k] = i
        
        numGrids = len(ID)
        self.log_time("dbsacn_grid-init grid", start_time)
        
        start_time = time.time()
        kdTree = construct_kdtree([k for k,v in grid.items()])
        self.log_time("dbsacn_grid-construct kdtree", start_time)

        # - Label core cells
        start_time = time.time()
        
        core_cells = parallel_find_in_grid_core_cell(grid, ID, self.min_samples, self.n_threads)
            
        core_cells = parallel_find_out_grid_core_cell(
            kdTree,
            eps_coverage,
            grid,
            self.min_samples,
            X,
            ID,
            self.eps,
            core_cells,
            self.n_threads
        )
        
        self.log_time("dbscan_grid-label core cells", start_time)
        
        # - Clustering
        start_time = time.time()
        labels = parallel_clustering(
            kdTree,
            eps_coverage,
            grid,
            self.min_samples,
            X,
            ID,
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