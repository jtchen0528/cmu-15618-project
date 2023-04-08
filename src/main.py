import numpy as np
import random
import argparse

import datasets
import models
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset [mnist]')
    parser.add_argument('--algorithm', type=str, default='kmeans', help='clustering algorithm [kmeans|dbscan|kmeans_sklearn]')
    parser.add_argument('--nthreads', type=int, default=1, help='number of threads to run')
    parser.add_argument('--clusters', type=int, default=10, help='number of clusters')
    parser.add_argument('--iteration', type=int, default=10, help='iteration of clustering')
    parser.add_argument('--output_dir', type=str, default="outputs", help='output result directory')
    parser.add_argument('--seed', type=int, default=15618, help='seed of randomness')
    parser.add_argument('--pca', type=int, default=0, help='PCA or not, [0|1]')
    parser.add_argument('--omp', type=int, default=0, help='OMP or not, [0|1]')

    args = parser.parse_args()
    dirname = f'{args.algorithm}_omp{args.omp}_{args.nthreads}threads_' + \
                f'{args.dataset}_pca{args.pca}_{args.clusters}cluster_{args.iteration}iter'
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, dirname)
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    (X_train, Y_train), (X_test, Y_test) = datasets.get_dataset(dataset = args.dataset, seed=args.seed, compress_data=args.pca)

    # Fit centroids to dataset
    model = models.get_model(algorithm=args.algorithm, n_clusters=args.clusters, max_iter=args.iteration, nthreads = args.nthreads, omp = args.omp)
    model.fit(X_train)
    # model.plot_centroids(output_dir)
    model.plot_2d_centroids(output_dir, X_test, Y_test)
    model.show_metrics(X_test, Y_test, output_dir)
    model.write_time_log(output_dir)

