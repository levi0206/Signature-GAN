from typing import Tuple
import torch
from torch import nn
from lib.network.resfnn import ResidualNN

class ArFNN(nn.Module):
    '''
    Sig-Wasserstein GANs for Conditional Time Series Generation
    https://arxiv.org/pdf/2006.05421.pdf
    Definition 23.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.network = ResidualNN(self.input_dim, self.output_dim , self.hidden_dims)
        self.latent_dim = latent_dim
    
    def forward(self, z: torch.Tensor, x_past: torch.Tensor):
        assert len(z.shape) == 3

        x_generated = list()
        batch_size = x_past.shape[0]
        window_size = z.shape[1]

        # For each lag, xt is concatenated with zt.
        for t in range(window_size):
            z_t = z[:, t:t + 1]
            x_in = torch.cat([z_t, x_past.reshape(batch_size, 1, -1)], dim=-1)
            x_gen = self.network(x_in)
            x_past = torch.cat([x_past[:, 1:], x_gen], dim=1)
            x_generated.append(x_gen)

        x_fake = torch.cat(x_generated, dim=1)
        return x_fake
    
    def sample(self, steps, x_past):
        z = torch.randn(x_past.size(0), steps, self.latent_dim).to(x_past.device)
        return self.forward(z, x_past)