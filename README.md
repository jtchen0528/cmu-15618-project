# cmu-15618-project

### Test Inputs
1. n_thread
2. algo: [kmeans|kmeans_omp|dbscan|dbscan_omp]
3. dataset: [mnist]
4. iteration: [(int)]
5. seed: [(int): 15618]
6. omp scheduler: [dynamic|static]

### Test Output (format output)
1. ```iterate_k_means``` time
2. cluster accuracy
3. data load time 

### To-do
1. mnist data load from local .npz  
2. centroid create method
    i. random (seed: 15618)
3. plot learned centroid
    i. derive from [kmeans_sklearn/kmeans.py](kmeans_sklearn/kmeans.py#L49)
4. accuracy calculation
    i. derive from [kmeans_sklearn/helpers.py](kmeans_sklearn/helpers.py)
5. Calculate time  
(6. Load data from memory or disk)

