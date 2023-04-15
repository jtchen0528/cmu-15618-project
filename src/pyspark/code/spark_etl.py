#!/usr/bin/env python3.8
# -*- coding: UTF-8 -*-

import sys
import pyspark
import os
from pyspark import StorageLevel

import datasets

import numpy as np
import random

def write_rdd(rdd, outfile, output_dir):
    """Write RDD to text file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = output_dir + '/' + outfile
    print(f'writing {filename}\n')
    with open(filename, 'w') as writer:
        if type(rdd) is not list:
            rdd = rdd.collect()
        for x in rdd:
            writer.write(str(x) + "\n")
        writer.close()


def euclidean_serial(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


def data_centroids_diff(datas):
    X_train = np.array([data[1] for data in datas])
    dists = np.sum([euclidean_serial(centroid, X_train) for centroid in centroids_bc], axis=0)
    dists = dists.tolist()
    return dists

def sum_centroids_dists(datas):
    dists = np.array([data[1] for data in datas])
    dists = np.sum(dists).tolist()
    return dists


def get_data_by_key(rdd, key):
    return rdd.filter(lambda x: x[0] == key).map(lambda x: x[1]).collect()


if __name__ == "__main__":
    """Drive main function."""  # noqa: E501
    dataset = sys.argv[1]
    algorithm = sys.argv[2]
    nthreads = int(sys.argv[3])
    clusters = int(sys.argv[4])
    iteration = int(sys.argv[5])
    output_dir = sys.argv[6]
    seed = int(sys.argv[7])
    pca = int(sys.argv[8])
    omp = int(sys.argv[9])

    dirname = f'{algorithm}_omp{omp}_{nthreads}threads_' + \
                f'{dataset}_pca{pca}_{clusters}cluster_{iteration}iter'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, dirname)
    os.makedirs(output_dir, exist_ok=True)

    conf = pyspark.SparkConf().setAppName("CommonCrawlProcessor")
    conf.set("spark.rpc.message.maxSize", "1024")
    sc = pyspark.SparkContext(conf=conf)

    random.seed(seed)
    np.random.seed(seed)

    (X_train, Y_train), (X_test, Y_test) = datasets.get_dataset(dataset = dataset, seed=seed, compress_data=pca)

    dataset_size = len(X_train)
    X_train_rdd = sc.parallelize(X_train, nthreads)

    X_train_rdd = X_train_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).partitionBy(lambda x: x % nthreads)
    write_rdd(X_train_rdd, "X_train_rdd", output_dir)

    init_centroid_idx = random.choice(range(dataset_size))[0]
    centroids = get_data_by_key(X_train_rdd, init_centroid_idx)
    centroids_bc = sc.broadcast(centroids)

    for _ in range(clusters-1):
        X_train_centroids_diffs = X_train_rdd.mapPartition(data_centroids_diff)
        X_train_centroids_diffs_sum = X_train_centroids_diffs.mapPartition(sum_centroids_dists).sum()
        new_centroids_probs = X_train_centroids_diffs.map(lambda x: x / X_train_centroids_diffs_sum).collect()
        new_centroid_idx = np.random.choice(range(dataset_size), size=1, p=new_centroids_probs)[0]
        centroids += []


    sc.stop()
