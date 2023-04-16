import numpy as np
from sklearn.datasets import load_iris as lr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_iris(path = "../data/iris_dataset.npz", seed = 15618, compress_data = False):
    
    if path != None:
        with np.load(path, allow_pickle=True) as f:
            X_train, Y_train = f["data"], f["target"]
    else:
        iris = lr()
        X_train = iris.data
        Y_train = iris.target
        
    # Normalize
    X_train = StandardScaler().fit_transform(X_train)
    
    if compress_data:
        pca = PCA(n_components=2, random_state=seed)
        X_train = pca.fit_transform(X_train)
        
    return (X_train, Y_train), (X_train, Y_train)