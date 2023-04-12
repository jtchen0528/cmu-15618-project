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

    X_train_rdd = sc.parallelize(X_train, nthreads)

    write_rdd(X_train_rdd, "X_train_rdd", output_dir)

    sc.stop()
