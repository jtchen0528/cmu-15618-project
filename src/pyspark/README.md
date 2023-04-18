## Setup PySpark Cluster
```
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3-pip
sudo apt install python3.8
sudo apt install awscli
aws configure
pip3 install flintrock boto3 sty
```

## Execute PySpark K-Means Clustering
```
./run.sh ${dataset} ${algorithm} ${nThreads} ${n_clusters} ${iterations} ${output_dir} ${random_seed} ${PCA}
```

```
./run.sh mnist kmeans 12 10 100 outputs 15618 0
```