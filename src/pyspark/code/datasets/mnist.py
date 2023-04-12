import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_mnist(path = "data/mnist.npz", seed = 15618, compress_data = False):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    X_train = x_train.reshape(len(x_train), -1)
    Y_train = y_train
    X_train = X_train.astype(float) / 255.
    X_test = x_test.reshape(len(x_test), -1)
    Y_test = y_test
    X_test = X_test.astype(float) / 255.

    if (compress_data):
        # Apply PCA to reduce the dimensions to 2
        SC = StandardScaler()
        X_train = SC.fit_transform(X_train)
        X_test = SC.fit_transform(X_test)
        pca = PCA(n_components=2, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.fit_transform(X_test)

    X_train = X_train.tolist()
    Y_train = Y_train.tolist()
    X_test = X_test.tolist()
    Y_test = Y_test.tolist()
    
    return (X_train, Y_train), (X_test, Y_test)