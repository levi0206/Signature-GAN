import json
import pickle

import torch
import numpy as np
import random

def sample_indices(dataset_size, batch_size, device):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False))
    if device == 'cuda':
        indices = indices.cuda()
    else:
        indices = indices
    
    return indices.long()

def to_numpy(x):
    return x.detach().cpu().numpy()

def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """

    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    elif filepath.endswith('json'):
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=4)
        return 0
    else:
        raise NotImplementedError()

    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)