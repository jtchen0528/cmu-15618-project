#%%
import numpy as np
import os
from keras.datasets import mnist
import helper
from sklearn import metrics

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label[0])

    return [np.array(cluster_label), centroids]

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids(X, n_clusters, seed):
    n_samples = X.shape[0]

    random_state = np.random.RandomState(seed)
    indices = random_state.permutation(n_samples)[:n_clusters]
    centers = X[indices]

    return np.array(centers)
#%%

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# preprocessing the images
# convert each image to 1 dimensional array
X = x_train.reshape(len(x_train),-1)
Y = y_train
#%%
#%%
# normalize the data to 0 - 1
X = X.astype(float) / 255.

n_clusters = 10

centroids = create_centroids(X, n_clusters, 15618)

total_iteration = 10

[cluster_label, new_centroids] = iterate_k_means(X, centroids, total_iteration)

inferred_labels = helper.infer_cluster_labels(n_clusters, cluster_label, Y)
predicted_Y = helper.infer_data_labels(cluster_label, inferred_labels)
#%%
print('Homogeneity: {}'.format(metrics.homogeneity_score(Y, cluster_label)))
#%%
print('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y)))
#%%
helper.plot_centroids(centroids, cluster_label, n_clusters, Y, "imgs")
print()

# %%
