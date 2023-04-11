from .mnist import *
from .iris import load_iris

def get_dataset(dataset = "mnist", seed = 15618, compress_data = True, filepath = None):
    print(f'Loading dataset [{dataset}]')
    if dataset == "mnist":
        return load_mnist(seed=seed, compress_data=compress_data)
    elif dataset == "iris":
        return load_iris(seed=seed, compress_data=compress_data)
    else:
        print(f'Dataset: [{dataset}] not support')
        return None
