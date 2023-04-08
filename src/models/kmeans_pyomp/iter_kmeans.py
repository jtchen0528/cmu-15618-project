import numpy as np
from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_set_num_threads, omp_get_thread_num, omp_get_num_threads, omp_get_wtime

import random

@njit
def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


def load_mnist(path = "data/mnist.npz"):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    X_train = x_train.reshape(len(x_train), -1)
    Y_train = y_train
    X_train = X_train.astype(float) / 255.
    X_test = x_test.reshape(len(x_test), -1)
    Y_test = y_test
    X_test = X_test.astype(float) / 255.

    return (X_train, Y_train), (X_test, Y_test)

(X_train, Y_train), (X_test, Y_test) = load_mnist()
random.seed(15618)
np.random.seed(15618)
n_clusters = 3
centroids = [random.choice(X_train)]
for _ in range(n_clusters-1):
    dists = np.sum([euclidean(centroid, X_train) for centroid in centroids], axis=0)
    dists /= np.sum(dists)
    new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]
    centroids += [X_train[new_centroid_idx]]

print((X_train[0] - centroids).shape)

centroids = np.array(centroids)

@njit
def iter_kmeans(X_train, n_clusters, centroids):

    data_size = X_train.shape[1]
    sorted_points = np.zeros((n_clusters, X_train.shape[1]))
    sorted_points_cnt = np.zeros((n_clusters, 1))
    sum = np.array([0, 0], dtype=np.int64)
    with openmp("parallel for reduction(+:sorted_points) reduction(+:sum)"):
        for i in range(len(X_train)):
            data_size = X_train[i].shape[0]
            dists = euclidean(X_train[i], centroids)
            centroid_idx = np.argmin(dists)
            sum += np.array([1, 1], dtype=np.int64)
            # tmp_sorted_point = np.zeros(data_size)
            # sorted_points += np.array(1)
            # sorted_points_cnt[centroid_idx] += 1

    # with openmp("parallel"):
    #     for i, cnt in enumerate(sorted_points_cnt):
    #         if cnt != 0:
    #             centroids[i] = sorted_points[i] / cnt

    # for i, centroid in enumerate(centroids):
    #     if np.isnan(centroid).any():
    #         centroids[i] = prev_centroids[i]
    print(sum)

n_clusters = np.int64(n_clusters)
iter_kmeans(X_train, n_clusters, centroids)
