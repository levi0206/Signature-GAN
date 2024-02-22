from typing import Tuple
import torch.nn as nn
import torch
import signatory

from lib.augmentations import apply_augmentations, get_number_of_channels_after_augmentations
from lib.network.resfnn import ResidualNN

def init_weights(m):
    '''
    Fill the input Tensor with values using a Xavier uniform distribution.
    https://proceedings.mlr.press/v9/glorot10a.html
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x
    
class LSTMGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int, init_fixed: bool = True):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)
        self.init_fixed = init_fixed

    def forward(self, batch_size: int, n_lags: int, device: str) -> torch.Tensor:
        z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim)).to(device)
        z[:, 0, :] *= 0  
        z = z.cumsum(1) 

        # Initial hidden state of LSTM
        # Check https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html for the shape of h0
        if self.init_fixed:
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

        # Initial cell state of LSTM
        # Check https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html for the shape of c0
        c0 = torch.zeros_like(h0)
        h1, _ = self.lstm(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags
        return x
    
def compute_multilevel_logsignature(brownian_path: torch.Tensor, time_brownian: torch.Tensor, time_u: torch.Tensor,
                                    time_t: torch.Tensor, depth: int):
    """

    Parameters
    ----------
    brownian_path: torch.Tensor
        Tensor of shape [batch_size, L, dim] where L is big enough so that we consider this 
    time_brownian: torch.Tensor
        Time evaluations of brownian_path
    time_u: torch.Tensor
        Time discretisation used to calculate logsignatures
    time_t: torch.Tensor
        Time discretisation of generated path
    depth: int
        depth of logsignature

    Returns
    -------
    multi_level_signature: torch.Tensor

    ind_u: List
        List of indices time_u used in the logsigrnn
    """
    logsig_channels = signatory.logsignature_channels(in_channels=brownian_path.shape[-1], depth=depth)

    multi_level_log_sig = []  # torch.zeros(brownian_path.shape[0], len(time_t), logsig_channels)

    u_logsigrnn = []
    last_u = -1
    for ind_t, t in enumerate(time_t[1:]):
        u = time_u[time_u < t].max()
        ind_low = torch.nonzero((time_brownian <= u).float(), as_tuple=False).max()
        if u != last_u:
            u_logsigrnn.append(u)
            last_u = u

        ind_max = torch.nonzero((time_brownian <= t).float(), as_tuple=False).max()
        interval = brownian_path[:, ind_low:ind_max + 1, :]
        multi_level_log_sig.append(signatory.logsignature(interval, depth=depth, basepoint=True))
    multi_level_log_sig = [torch.zeros_like(multi_level_log_sig[0])] + multi_level_log_sig

    return multi_level_log_sig, u_logsigrnn

class FeedForwardNN(nn.Module):
    """Same as ResidualNN but with PReLU"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int]):
        super().__init__()
        blocks = []
        block_input_dim = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(nn.Linear(block_input_dim, hidden_dim))
            blocks.append(nn.PReLU())
            block_input_dim = hidden_dim
        blocks.append(nn.Linear(block_input_dim, output_dim))
        self.network = nn.Sequential(*blocks)
        self.output_dim = output_dim

    def forward(self, *args):
        x = torch.cat(args, -1)
        out = self.network(x)
        return out
    
class LogSigRNNGenerator(GeneratorBase):
    def __init__(self, input_dim, output_dim, augmentations, depth, hidden_dim, len_noise=1000,
                 len_interval_u=50, init_fixed: bool = True):

        super(LogSigRNNGenerator, self).__init__(input_dim, output_dim)
        input_dim_rnn = get_number_of_channels_after_augmentations(input_dim, augmentations)

        logsig_channels = signatory.logsignature_channels(in_channels=input_dim_rnn, depth=depth)

        self.depth = depth
        self.augmentations = augmentations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.len_noise = len_noise
        self.time_brownian = torch.linspace(0, 1,
                                            self.len_noise)  # len_noise is high enough so that we can consider this as a continuous brownian motion
        self.time_u = self.time_brownian[::len_interval_u]  # ([0.0000, 0.0501, 0.1001, 0.1502, 0.2002, 0.2503, 0.3003, 0.3504, 0.4004, 0.4505, 0.5005, 0.5506, 0.6006, 0.6507, 0.7007, 0.7508, 0.8008, 0.8509, 0.9009, 0.9510])
        # self.time_t = torch.linspace(0,1,window_size)

        # definition of LSTM + linear at the end
        self.rnn = nn.Sequential(
            FeedForwardNN(input_dim=hidden_dim + logsig_channels,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)

        # neural network to initialise h0 from the LSTM
        self.initial_nn = nn.Sequential(
            ResidualNN(input_dim, hidden_dim, [hidden_dim, hidden_dim]),
            nn.Tanh()
        )
        self.initial_nn.apply(init_weights)
        self.init_fixed = init_fixed

    def forward(self, batch_size: int, window_size: int, device: str, ):
        time_t = torch.linspace(0, 1, window_size).to(device)
        # time_t: torch.Size([72]) = [n_lags]
        # time_brownian: torch.Size([1000]) = [gen_config.len_noise]

        # z: torch.Size([1024, 1000, 5])
        z = torch.randn(batch_size, self.len_noise, self.input_dim, device=device)
        
        # h is the time step of the Brownian motion. All cell values same
        # h: torch.Size([1, 999, 5]) = [1, gen_config.len_noise - 1 , gen_config.input_dim]
        h = (self.time_brownian[1:] - self.time_brownian[:-1]).reshape(1, -1, 1).repeat(1, 1, self.input_dim)
        h = h.to(device)

        z[:, 1:, :] *= torch.sqrt(h)
        z[:, 0, :] *= 0  # first point is fixed

        brownian_path = z.cumsum(1)
        # brownian_path = z

        y = apply_augmentations(brownian_path, self.augmentations)
        y_logsig, u_logsigrnn = compute_multilevel_logsignature(brownian_path=y,
                                                                time_brownian=self.time_brownian.to(device),
                                                                time_u=self.time_u.to(device), time_t=time_t.to(device),
                                                                depth=self.depth)
        # [y_logsig] is a list, length 72 = n_lags.
        # Each element size in [y_logsig]: torch.Size([1024, 91]) = [batch_size, log_signature_len]

        # [u_logsigrnn] is a list of 20 values which are timestamps (the same as self.time_u)
        # [tensor(0., device='cuda:0'), tensor(0.0501, device='cuda:0'), tensor(0.1001, device='cuda:0'), ... , tensor(0.9510, device='cuda:0')]
        
        # becomes [tensor(0., device='cuda:0'), ..., [tensor(1., device='cuda:0')]
        u_logsigrnn.append(time_t[-1]) # essentially adding the final timestamp

        if self.init_fixed:  # gen_config
            h0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        else:
            z0 = torch.randn(batch_size, self.input_dim, device=device)
            h0 = self.initial_nn(z0)
        last_h = h0
        x = torch.zeros(batch_size, window_size, self.output_dim, device=device)
        for idx, (t, y_logsig_) in enumerate(zip(time_t, y_logsig)):
            h = self.rnn(torch.cat([last_h, y_logsig_], -1))
            if t >= u_logsigrnn[0]:
                del u_logsigrnn[0]
                last_h = h  # this is why the yellow hidden nodes have periodically in my depiction
            x[:, idx, :] = self.linear(h)

        assert x.shape[1] == window_size
        return x