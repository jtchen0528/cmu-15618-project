from .mnist import *

def get_dataset(dataset = "mnist", seed = 15618, compress_data = True, filepath = None):
    print(f'Loading dataset [{dataset}]')
    if dataset == "mnist":
        return load_mnist(seed=seed, compress_data=compress_data)
    else:
        print(f'Dataset: [{dataset}] not support')
        return None
