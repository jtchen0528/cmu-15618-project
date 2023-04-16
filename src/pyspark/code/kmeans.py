#!/usr/bin/env python3.8
# -*- coding: UTF-8 -*-

import sys
import pyspark
import os
from pyspark import StorageLevel

import datasets

import numpy as np
import random
import time
import json

def write_rdd(rdd, outfile, output_dir):
    """Write RDD to text file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = output_dir + '/' + outfile
    print(f'writing {filename}\n')
    with open(filename, 'w') as writer:
        if type(rdd) is not list:
            try:
                rdd = rdd.collect()
            except:
                pass
        if type(rdd) is list:
            for x in rdd:
                writer.write(str(x) + "\n")
        else:
            writer.write(str(rdd) + "\n")
        writer.close()


def euclidean_serial(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))


def data_centroids_diff(datas):
    X_train = []
    ids = []
    for data in datas:
        ids.append(data[0])
        X_train.append(data[1])
    X_train = np.array(X_train)
    ids = np.array(ids)
    dists = np.sum([euclidean_serial(centroid, X_train) for centroid in centroids_bc.value], axis=0)
    dists = dists.tolist()
    dists_with_id = []
    i = 0
    for dist in dists:
        dists_with_id.append((ids[i], dist))
        i += 1
    
    return dists_with_id

def sum_centroids_dists(datas):
    dists = np.array(datas)
    dists = np.sum(dists).tolist()
    return dists


def get_data_by_key(rdd, key):
    return rdd.filter(lambda x: x[0] == key).collect()[0]


def iter_kmeans_serial(datas):
    centroids = np.array(centroids_bc.value)
    sorted_points = []
    for data in datas:
        dists = euclidean_serial(centroids, np.array(data[1]))
        centroid_idx = np.argmin(dists)
        sorted_points.append((data[0], centroid_idx))
    
    return sorted_points


def calculate_new_clusters(datas):
    datapoints = np.array([data[1] for data in datas])
    datapoints_mean = np.mean(datapoints, axis=0).tolist()
    return [datapoints_mean]


def log_time(time_log, entry_name, start_time):
    time_log[entry_name] = time.time() - start_time
    return time_log


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

    dirname = f'{algorithm}_{nthreads}threads_' + \
                f'{dataset}_pca{pca}_{clusters}cluster_{iteration}iter'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, dirname)
    os.makedirs(output_dir, exist_ok=True)

    time_log = {}

    conf = pyspark.SparkConf().setAppName("CommonCrawlProcessor")
    conf.set("spark.rpc.message.maxSize", "1024")
    sc = pyspark.SparkContext(conf=conf)

    random.seed(seed)
    np.random.seed(seed)

    (X_train, Y_train), (X_test, Y_test) = datasets.get_dataset(dataset = dataset, seed=seed, compress_data=pca)

    dataset_size = len(X_train)
    X_train_rdd = sc.parallelize(X_train, nthreads)

    X_train_rdd = X_train_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).partitionBy(nthreads, partitionFunc=lambda x: x % nthreads)
    X_train_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    start_time = time.time()

    init_centroid_idx = random.choice(range(dataset_size))

    centroids = get_data_by_key(X_train_rdd, init_centroid_idx)
    centroids = [centroids[1]]
    prev_centroids = np.array(centroids)
    centroids_bc = sc.broadcast(centroids)

    for _ in range(clusters-1):
        X_train_centroids_diffs = X_train_rdd.mapPartitions(data_centroids_diff)
        X_train_centroids_diffs_sum = X_train_centroids_diffs.map(lambda x: x[1]).sum()
        new_centroids_probs = X_train_centroids_diffs.map(lambda x: (x[0], x[1] / X_train_centroids_diffs_sum)).sortByKey().collect()
        new_centroids_probs = [prob[1] for prob in new_centroids_probs]
        new_centroid_idx = np.random.choice(range(dataset_size), size=1, p=new_centroids_probs)[0]
        new_centroid = get_data_by_key(X_train_rdd, new_centroid_idx)
        new_centroids = centroids_bc.value + [new_centroid[1]]
        centroids_bc = sc.broadcast(new_centroids)

    time_log = log_time(time_log, "centroid_init", start_time)

    current_iteration = 0
    centroids = np.array(centroids_bc.value)

    start_time = time.time()

    while np.not_equal(centroids, prev_centroids).any() and current_iteration < iteration:
        
        iter_data_start_time = time.time()
        
        sorted_points = X_train_rdd.mapPartitions(iter_kmeans_serial)
        
        time_log = log_time(time_log, f'iter_data_{current_iteration}', iter_data_start_time)

        iter_start_time = time.time()
        prev_centroids_list = centroids_bc.value
        prev_centroids = np.array(prev_centroids_list)
        new_centroids = []
        for i in range(clusters):
            cluster_datapoints_indices = sorted_points.filter(lambda x: x[1] == i).map(lambda x: x[0]).collect()

            if len(cluster_datapoints_indices) > 0:
                cluster_datapoints = X_train_rdd.filter(lambda x: x[0] in cluster_datapoints_indices)
                cluster_datapoints_means = cluster_datapoints.mapPartitions(calculate_new_clusters).collect()
                new_centroid = np.mean(np.array(cluster_datapoints_means), axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(prev_centroids_list[i])
        centroids_bc = sc.broadcast(new_centroids)
        centroids = np.array(new_centroids)

        time_log = log_time(time_log, f'iter_{current_iteration}', iter_start_time)

        current_iteration += 1

    time_log = log_time(time_log, "total", start_time)

    centroids_np = np.array(centroids_bc.value)
    np.savez(os.path.join(output_dir, f'centroids_final'), centroids_np)

    with open(os.path.join(output_dir, "time_log.txt"), "w") as f:
        f.write(json.dumps(time_log, indent = 4))

    sc.stop()
