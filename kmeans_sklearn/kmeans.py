import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.cluster import KMeans
import helpers
from sklearn import metrics
import os
import math

os.makedirs("imgs", exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))
print('Testing Data: {}'.format(x_test.shape))
print('Testing Labels: {}'.format(y_test.shape))

# preprocessing the images
# convert each image to 1 dimensional array
X = x_train.reshape(len(x_train),-1)
Y = y_train

# normalize the data to 0 - 1
X = X.astype(float) / 255.

clusters = [9, 16, 36, 64, 144, 256]

# test different numbers of clusters
for n_clusters in clusters:
    estimator = KMeans(n_clusters = n_clusters)
    estimator.fit(X)
    
    # print cluster metrics
    helpers.calculate_metrics(estimator, X, Y)
    
    # determine predicted labels
    cluster_labels = helpers.infer_cluster_labels(estimator, Y)
    predicted_Y = helpers.infer_data_labels(estimator.labels_, cluster_labels)
    
    # calculate and print accuracy
    print('Accuracy: {}\n'.format(metrics.accuracy_score(Y, predicted_Y)))

    # record centroid values
    centroids = estimator.cluster_centers_

    # reshape centroids into images
    images = centroids.reshape(n_clusters, 28, 28)
    images *= 255
    images = images.astype(np.uint8)

    # determine cluster labels
    cluster_labels = helpers.infer_cluster_labels(estimator, Y)

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
    fig.savefig(f'imgs/c{n_clusters}.png')
