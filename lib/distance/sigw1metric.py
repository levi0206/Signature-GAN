from typing import Tuple, Optional

import signatory
import numpy as np
import torch
from torch import autograd
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import math

from lib.augmentations import apply_augmentations
# from lib.trainers.base import Base
from lib.utils import sample_indices
from torch import optim

def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
    x_path_augmented = apply_augmentations(x_path, augmentations)
    
    # Monte Carlo: expectation -> mean
    expected_signature = signatory.signature(x_path_augmented, depth=depth).mean(0)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            expected_signature[count:count + dim ** (i + 1)] = expected_signature[
                                                               count:count + dim ** (i + 1)] * math.factorial(i + 1)
            count = count + dim ** (i + 1)
    return expected_signature


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()


def masked_rmse(x, y, mask_rate, device):
    mask = torch.FloatTensor(x.shape[0]).to(device).uniform_() > mask_rate
    mask = mask.int()
    return ((x - y).pow(2) * mask).mean().sqrt()

class SigW1Metric:
    def __init__(self, depth: int, x_real: torch.Tensor, mask_rate: float, augmentations: Optional[Tuple] = (),
                 normalise: bool = True):
        assert len(x_real.shape) == 3, \
            'Path needs to be 3-dimensional. Received %s dimension(s).' % (len(x_real.shape),)

        self.augmentations = augmentations
        self.depth = depth
        self.window_size = x_real.shape[1]
        self.mask_rate = mask_rate
        self.normalise = normalise
        self.expected_signature_mu = compute_expected_signature(x_real, depth, augmentations, normalise)

    def __call__(self, x_path_nu: torch.Tensor):
        """
        Computes the SigW1 metric.\n
        Equation (4) in 2111.01207
        """
        device = x_path_nu.device
        batch_size = x_path_nu.shape[0]
        # expected_signature_nu1 = compute_expected_signature(x_path_nu[:batch_size//2], self.depth, self.augmentations)
        # expected_signature_nu2 = compute_expected_signature(x_path_nu[batch_size//2:], self.depth, self.augmentations)
        # y = self.expected_signature_mu.to(device)
        # loss = (expected_signature_nu1-y)*(expected_signature_nu2-y)
        # loss = loss.sum()
        expected_signature_nu = compute_expected_signature(x_path_nu, self.depth, self.augmentations, self.normalise)
        loss = rmse(self.expected_signature_mu.to(device), expected_signature_nu)
        # loss = masked_rmse(self.expected_signature_mu.to(
        #    device), expected_signature_nu, self.mask_rate, device)
        return loss