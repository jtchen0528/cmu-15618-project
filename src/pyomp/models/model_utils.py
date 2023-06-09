from .kmeans_serial import *
from .kmeans_sklearn import *
from .kmeans_pyomp import *
from .kmeans_inference import *
from .DBSCAN import DBSCAN

def get_model(algorithm="kmeans", n_clusters=10, max_iter=10, nthreads=1, omp=False, eps=0.5, min_samples=5, init_centroids_path = ""):
    model = None
    prefix = algorithm[:6]
    if prefix == "kmeans":
        print(f'Loading model [{algorithm}] with [{n_clusters}] clusters, [{max_iter}] iterations, and [{nthreads}] threads')
        if algorithm == "kmeans_serial":
            model = KMeans_serial(n_clusters=n_clusters, max_iter=max_iter)
        if algorithm == "kmeans_inference":
            centroids = np.load(init_centroids_path)
            centroids = centroids['arr_0']
            model = KMeans_inference(n_clusters=n_clusters, max_iter=max_iter, centroids=centroids)
        elif algorithm == "kmeans_sklearn":
            model = KMeans_sklearn(n_clusters=n_clusters, max_iter=max_iter)
        elif algorithm == "kmeans":
            model = KMeans_pyomp(n_clusters=n_clusters, max_iter=max_iter, n_threads=nthreads, omp = omp)
        else:
            print(f'Algorithm: [{algorithm}] not support')
    elif prefix == "dbscan":
        # move print to constructor
        if algorithm == "dbscan_grid":
            model = DBSCAN(model = 'grid', eps=eps, min_samples=min_samples)
        elif algorithm == "dbscan_grid_mp":
            model = DBSCAN(model = 'grid_mp', eps=eps, min_samples=min_samples, n_threads=nthreads, omp=omp)
        elif algorithm == "dbscan_sklearn":
            model = DBSCAN(model = 'sklearn', eps=eps, min_samples=min_samples)
        else:
            print(f'Algorithm: [{algorithm}] not support')
    else:
        print(f'Algorithm: [{algorithm}] not support')
    return model