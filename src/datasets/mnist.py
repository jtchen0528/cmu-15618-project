import numpy as np

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