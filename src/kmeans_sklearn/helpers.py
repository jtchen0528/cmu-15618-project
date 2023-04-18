import numpy as np
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import os

def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

def calculate_metrics(estimator, data, labels):

    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))

def plot_centroids(estimator, n_clusters, Y, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # record centroid values
    centroids = estimator.cluster_centers_

    # reshape centroids into images
    images = centroids.reshape(n_clusters, 28, 28)
    images *= 255
    images = images.astype(np.uint8)

    # determine cluster labels
    cluster_labels = infer_cluster_labels(estimator, Y)

    edge = int(math.sqrt(n_clusters))

    # create figure with subplots using matplotlib.pyplot
    fig, axs = plt.subplots(edge, edge, figsize = (20, 20))
    plt.gray()

    # loop through subplots and add centroid images
    for i, ax in enumerate(axs.flat):
        
        # determine inferred label using cluster_labels dictionary
        for key, value in cluster_labels.items():
            if i in value:
                ax.set_title('Inferred Label: {}'.format(key))
        
        # add image to subplot
        ax.matshow(images[i])
        ax.axis('off')
        
    # display the figure
    fig.savefig(f'{output_dir}/c{n_clusters}.png')
