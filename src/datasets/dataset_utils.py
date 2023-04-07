from .mnist import *

def get_dataset(dataset = "mnist", filepath = None):
    print(f'Loading dataset [{dataset}]')
    if dataset == "mnist":
        return load_mnist()
    else:
        print(f'Dataset: [{dataset}] not support')
        return None
