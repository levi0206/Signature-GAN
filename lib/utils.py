import json
import os
import pickle
from datetime import datetime, timedelta
import tqdm

import torch
import numpy as np
import random

### Copy, no use
def get_config_path(config_type, name):
    return os.path.join('configs', config_type, name + '.json')


def get_sigwgan_experiment_dir(dataset, generator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, seed=seed)

def get_experiment_dir(dataset, generator, discriminator, gan, seed):
    return './numerical_results/{dataset}/{gan}_{generator}_{discriminator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, discriminator=discriminator, seed=seed)
###

def sample_indices(dataset_size, batch_size, device):
    '''
    Use np.random.choice to sample data: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    '''
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
    '''
    Save an object.
    '''
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
    '''
    Load an object.
    '''
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
    
# Probably no need    
def rolling_period_resample(dataset, period, n_lags):
    period_dict = {
        'd': 0,
        'h': 0
    }
    assert period[-1] in period_dict.keys()
    period_dict[period[-1]] = int(period[:-1])
    time_period = timedelta(days=period_dict['d'], hours=period_dict['h'])
    time_clip = time_period / n_lags
    dataset_length = len(dataset)

    # print('length', dataset_length)

    def find_data(start_i, target_t):
        i_point = 0
        kk = 0
        while start_i + i_point < dataset_length - 1:
            if dataset[start_i + i_point][0] <= target_t <= dataset[start_i + i_point + 1][0]:
                return start_i + i_point
            else:
                if dataset[start_i + i_point][0] > target_t:
                    print(f'find {target_t} from index {start_i}={dataset[start_i]}. '
                          f'current {dataset[start_i + i_point][0]}')
                    kk += 1
                    if kk > 5:
                        exit()
                i_point += 1

    rolled_dataset = []
    rolled_data_idx = [0] * (n_lags - 1)
    for i, data in enumerate(tqdm(dataset)):
        start_time = datetime.fromtimestamp(int(data[0]))
        end_time = (start_time + time_period).timestamp()
        if dataset[-1][0] < end_time:
            break

        rolled_data = [data]
        time_point = (start_time + time_clip).timestamp()

        data_idx = 0
        for k, idx in enumerate(rolled_data_idx):
            idx = max(idx, data_idx)
            assert dataset[idx][0] < time_point
            data_idx = find_data(idx, time_point)
            rolled_data.append(dataset[data_idx])
            rolled_data_idx[k] = data_idx
            time_point = (start_time + time_clip * len(rolled_data)).timestamp()

        # print(rolled_data_idx)
        rolled_dataset.append(np.stack(rolled_data))

    return np.stack(rolled_dataset)