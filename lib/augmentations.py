import torch
from dataclasses import dataclass
from typing import List, Tuple

import signatory

def get_time_vector(batch_size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(batch_size, 1, 1)

def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation. 
    See "A Primer on the Signature Method in Machine Learning"
    """
    # 0-th dim is batch
    repeat = torch.repeat_interleave(x, repeats=2, dim=1) 
    lead_lag = torch.cat([repeat[:, :-1], repeat[:, 1:]], dim=2)
    return lead_lag

def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_repeat = torch.repeat_interleave(t, repeats=3, dim=1)
    x_repeat = torch.repeat_interleave(x, repeats=3, dim=1)
    time_lead_lag = torch.cat([
        t_repeat[:, 0:-2],
        x_repeat[:, 1:-1],
        x_repeat[:, 2:],
    ], dim=2)
    return time_lead_lag

def sig_normal(sig: torch.Tensor, normalize=False):
    if normalize == False:
        return sig.mean(0)
    elif normalize == True:
        mu = sig.mean(0)
        sigma = sig.std(0)
        sig = (sig-mu)/sigma
        return sig
    
def get_number_of_channels_after_augmentations(input_dim, augmentations):
    x = torch.zeros(1, 10, input_dim)
    y = apply_augmentations(x, augmentations)
    return y.shape[-1]

@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')
    
@dataclass
class AddTime(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)
    
@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        if self.with_time:
            return lead_lag_transform_with_time(x)
        else:
            return lead_lag_transform(x)

def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y

AUGMENTATIONS = {'AddTime': AddTime, 'LeadLag': LeadLag} 