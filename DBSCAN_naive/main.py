from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data into a 2D array of shape (n_samples, n_features)
X_train = x_train.reshape((x_train.shape[0], -1))
X_test = x_test.reshape((x_test.shape[0], -1))

# Apply PCA to reduce the dimensions to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Apply DBSCAN to cluster the data
dbscan = DBSCAN(eps=10, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_train_pca)

# Create a scatter plot of the clustered data
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=dbscan_labels)
plt.title('DBSCAN Clustering of MNIST Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('dbscan_mnist_clusters.png')
plt.show()