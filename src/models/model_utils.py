from .kmeans_serial import *
from .kmeans_sklearn import *
from .kmeans_pyomp import *
from .DBSCAN import DBSCAN

def get_model(algorithm = "kmeans", n_clusters=10, max_iter=10, nthreads = 1, omp = False):
    model = None
    print(f'Loading model [{algorithm}] with [{n_clusters}] clusters, [{max_iter}] iterations, and [{nthreads}] threads')
    if algorithm == "kmeans_serial":
        model = KMeans_serial(n_clusters=n_clusters, max_iter=max_iter)
    elif algorithm == "kmeans_sklearn":
        model = KMeans_sklearn(n_clusters=n_clusters, max_iter=max_iter)
    elif algorithm == "kmeans":
        model = KMeans_pyomp(n_clusters=n_clusters, max_iter=max_iter, n_threads=nthreads, omp = omp)
    elif algorithm == "dbscan_grid":
        model = DBSCAN(model = 'grid')
    elif algorithm == "dbscan_sklearn":
        model = DBSCAN(model = 'sklearn')
    else:
        print(f'Algorithm: [{algorithm}] not support')
    return model