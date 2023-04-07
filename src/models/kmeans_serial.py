import numpy as np
import matplotlib.pyplot as plt
import random

import os
import math
from sklearn import metrics
from tqdm import tqdm
import time
import json

def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans_serial:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.time_log = {}
        print(f'Kmeans, [{self.n_clusters}] clusters, [{self.max_iter}] iterations, Loaded.')

    def log_time(self, entry_name, start_time):
        self.time_log[entry_name] = time.time() - start_time

    def fit(self, X_train):
        start_time = time.time()
        print("[TRAIN] Centroids initialization")
        self.centroids = [random.choice(X_train)]
        prev_centroids = [random.choice(X_train)]

        for _ in range(self.n_clusters-1):
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            dists /= np.sum(dists)
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]
            self.centroids += [X_train[new_centroid_idx]]
        self.log_time("centroid_init", start_time)

        start_time = time.time()
        print("[TRAIN] Start training")
        iteration = 0
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            iter_start_time = time.time()
            sorted_points = [[] for _ in range(self.n_clusters)]
            batch_bar = tqdm(total=len(X_train), dynamic_ncols=True, leave=False, position=0, desc=f'Iter: {iteration}', ncols=5) 
            iter_data_start_time = time.time()
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
                batch_bar.update()
            batch_bar.close()
            self.log_time(f'iter_data_{iteration}', iter_data_start_time)

            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
            self.log_time(f'iter_{iteration}', iter_start_time)

        self.log_time("total", start_time)


    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
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