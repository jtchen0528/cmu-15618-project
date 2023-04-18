# ref: https://github.com/sharmaroshan/MNIST-Using-K-means/blob/master/KMeans%20Clustering%20for%20Imagery%20Analysis%20(Jupyter%20Notebook).ipynb

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

    helpers.plot_centroids(estimator, n_clusters, Y, "imgs")
