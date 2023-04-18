## Setup Environment
1. Create Conda environment
    ```
    conda create -n pl-project python=3.7
    conda activate pl-project
    conda install drtodd13::numba cffi -c conda-forge --override-channels
    ```
2. Follow installation instructions on [Python-for-HPC/PyOMP](https://github.com/Python-for-HPC/PyOMP)
3. ```pip install -r requirements.txt```

## Execute PyOMP K-Means Clustering
```
python main.py \
    --dataset mnist \
    --algorithm kmeans \
    --nthreads 12 \
    --clusters 10 \
    --iteration 100 \
    --output_dir outputs \
    --seed 15618 \
    --pca 0 \
    --omp 1
```