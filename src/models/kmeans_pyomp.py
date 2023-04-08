# ref: https://github.com/turnerluke/ML-algos/blob/main/k_means/k_means.py

import numpy as np
import matplotlib.pyplot as plt
import random

import os
import math
from sklearn import metrics
from tqdm import tqdm
import time
import json
import seaborn as sns

from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_set_num_threads, omp_get_thread_num, omp_get_num_threads, omp_get_wtime

@njit
def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


@njit
def iter_kmeans(X_train, centroids, n_threads):
    omp_set_num_threads(n_threads)
    sorted_points = np.zeros(len(X_train))
    with openmp("parallel for"):
        for i in range(len(X_train)):
            dists = euclidean(X_train[i], centroids)
            centroid_idx = np.argmin(dists)
            sorted_points[i] = centroid_idx
    return sorted_points


@njit
def calculate_new_clusters(X_train, n_clusters, sorted_points, centroids, n_threads):
    omp_set_num_threads(n_threads)
    with openmp("parallel for"):
        for i in range(n_clusters):
            added = 0
            cluster_datapoints = np.zeros(centroids[i].shape, dtype=np.float64)
            with openmp("parallel for reduction(+:added)"):
                for idx in range(len(sorted_points)):
                    if sorted_points[idx] == i:
                        added += 1
                        cluster_datapoints += X_train[idx]
            if added:
                with openmp("parallel for"):
                    for idx in range(len(cluster_datapoints)):
                        centroids[i][idx] = cluster_datapoints[idx] / np.float64(added)
    return centroids


def iter_kmeans_serial(X_train, centroids):
    sorted_points = np.zeros(len(X_train))
    for i in range(len(X_train)):
        dists = euclidean_serial(X_train[i], centroids)
        centroid_idx = np.argmin(dists)
        sorted_points[i] = centroid_idx
    return sorted_points

def calculate_new_clusters_serial(X_train, n_clusters, sorted_points, centroids):
    for i in range(n_clusters):
        added = 0
        cluster_datapoints = np.zeros(centroids[i].shape, dtype=np.float64)
        for idx in range(len(sorted_points)):
            if sorted_points[idx] == i:
                added += 1
                cluster_datapoints += X_train[idx]
        if added:
            for idx in range(len(cluster_datapoints)):
                centroids[i][idx] = cluster_datapoints[idx] / np.float64(added)
    return centroids

def euclidean_serial(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans_pyomp:
    def __init__(self, n_clusters=8, max_iter=300, n_threads = 1, omp = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.time_log = {}
        self.n_threads = n_threads
        self.omp = omp
        print(f'Kmeans, [{self.n_clusters}] clusters, [{self.max_iter}] iterations, Loaded.')

    def log_time(self, entry_name, start_time):
        self.time_log[entry_name] = time.time() - start_time

    def fit(self, X_train):
        start_time = time.time()
        print("[TRAIN] Centroids initialization")
        self.centroids = [random.choice(X_train)]
        prev_centroids = [random.choice(X_train)]

        for _ in range(self.n_clusters-1):
            dists = np.sum([euclidean_serial(centroid, X_train) for centroid in self.centroids], axis=0)
            dists /= np.sum(dists)
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]
            self.centroids += [X_train[new_centroid_idx]]
        self.log_time("centroid_init", start_time)

        self.centroids = np.array(self.centroids)
        prev_centroids = np.array(prev_centroids)

        start_time = time.time()
        print("[TRAIN] Start training")
        iteration = 0
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:

            prev_centroids = np.copy(self.centroids)
            iter_start_time = time.time()
            iter_data_start_time = time.time()

            if self.omp is False:
                sorted_points = iter_kmeans_serial(X_train, self.centroids)
            else:
                sorted_points = iter_kmeans(X_train, self.centroids, self.n_threads)

            self.log_time(f'iter_data_{iteration}', iter_data_start_time)

            if self.omp is False:
                self.centroids = calculate_new_clusters_serial(X_train, self.n_clusters, sorted_points, self.centroids)
            else:
                self.centroids = calculate_new_clusters(X_train, self.n_clusters, sorted_points, self.centroids, self.n_threads)

            iteration += 1
            self.log_time(f'iter_{iteration}', iter_start_time)

        self.log_time("total", start_time)


    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean_serial(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return np.array(centroids), np.array(centroid_idxs)
    
    def plot_centroids(self, output_dir):
        print("[PLOT] Start plotting centroid")

        os.makedirs(output_dir, exist_ok=True)

        images = np.array(self.centroids).reshape(self.n_clusters, 28, 28)
        images *= 255
        images = images.astype(np.uint8)

        edge = int(math.sqrt(self.n_clusters))

        fig, axs = plt.subplots(edge, edge, figsize = (20, 20))
        plt.gray()

        for i, ax in enumerate(axs.flat):
            ax.matshow(images[i])
            ax.axis('off')
            
        fig.savefig(f'{output_dir}/centroids.png')


    def plot_2d_centroids(self, output_dir, X_train, Y_train):
        print("[PLOT] Start plotting centroid")

        os.makedirs(output_dir, exist_ok=True)
        class_centers, classification = self.evaluate(X_train)

        sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=Y_train,
                style=classification,
                palette="deep",
                legend=None
                )
        plt.plot([x[0] for x in self.centroids],
                [y[1] for y in self.centroids],
                '+',
                markersize=10,
                )
        plt.title("Centroids and Test data")
            
        plt.savefig(f'{output_dir}/centroids.png')

    def infer_cluster_labels(self, classification, Y):

        inferred_labels = {}

        for i in range(self.n_clusters):

            labels = []
            index = np.where(classification == i)

            labels.append(Y[index])

            if len(labels[0]) == 1:
                counts = np.bincount(labels[0])
            else:
                counts = np.bincount(np.squeeze(labels))

            if np.argmax(counts) in inferred_labels:
                inferred_labels[np.argmax(counts)].append(i)
            else:
                inferred_labels[np.argmax(counts)] = [i]

        return inferred_labels  


    def infer_data_labels(self, Y_labels, inferred_labels):
        predicted_labels = np.zeros(len(Y_labels)).astype(np.uint8)
        
        for i, cluster in enumerate(Y_labels):
            for key, value in inferred_labels.items():
                if cluster in value:
                    predicted_labels[i] = key
                    
        return predicted_labels

    def show_metrics(self, X, Y, output_dir):
        # View results
        print("[RESULT]")
        class_centers, classification = self.evaluate(X)

        inferred_labels = self.infer_cluster_labels(classification, Y)
        predicted_Y_train = self.infer_data_labels(classification, inferred_labels)

        print('Homogeneity: {}'.format(metrics.homogeneity_score(Y, classification)))
        print('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y_train)))

        with open(os.path.join(output_dir, "accuracy.txt"), "w") as f:
            f.write('Homogeneity: {}\n'.format(metrics.homogeneity_score(Y, classification)))
            f.write('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y_train)))

    def write_time_log(self, output_dir):
        with open(os.path.join(output_dir, "time_log.txt"), "w") as f:
            f.write(json.dumps(self.time_log, indent = 4))