from .kmeans import *

def get_model(algorithm = "kmeans", n_clusters=10, max_iter=10, nthreads = 1):
    model = None
    print(f'Loading model [{algorithm}] with [{n_clusters}] clusters, [{max_iter}] iterations, and [{nthreads}] threads')
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    else:
        print(f'Algorithm: [{algorithm}] not support')
    return model