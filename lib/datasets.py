import os
import glob
import pandas as pd
import random
import yfinance as yf
from fbm import fbm, MBM
import tqdm

import torch
import numpy as np
from lib.utils import sample_indices

def train_test_split(
        x: torch.Tensor,
        train_test_ratio: float,
        device: str
):
    size = x.shape[0]
    train_set_size = int(size * train_test_ratio)

    indices_train = sample_indices(size, train_set_size, device)
    indices_test = torch.LongTensor([i for i in range(size) if i not in indices_train])

    x_train = x[indices_train]
    x_test = x[indices_test]
    return x_train, x_test

def download_stock_price(
        ticker : str,
        start : str = '2000-01-01',
        end : str = '2023-12-31',
        interval: str = '1mo',
):
    dataframe = yf.download(ticker, start=start, end=end, interval=interval)
    dataframe.to_csv(f"./datasets/stock prices/{ticker}_{interval}.csv")
    return

def get_rBergomi_paths(hurst=0.25, size=2200, n_lags=100, maturity=1, xi=0.5, eta=0.5):
    r"""
    Paths of Rough stochastic volatility model for an asset price process S_t of the form

    dS_t = \sqrt(V_t) S_t dZ_t
    V_t := \xi * exp(\eta * W_t^H - 0.5*\eta^2*t^{2H})

    where W_t^H denotes the Riemann-Liouville fBM given by

    W_t^H := \int_0^t K(t-s) dW_t,  K(r) := \sqrt{2H} r^{H-1/2}

    with W_t,Z_t correlated brownian motions (I'm actually considering \rho=0)

    Parameters
    ----------
    hurst: float,
    size: int
        size of the dataset
    n_lags: int
        Number of timesteps in the path
    maturity: float
        Final time. Should be a value in [0,1]
    xi: float
    eta: float

    Returns
    -------
    dataset: np.array
        array of shape (size, n_lags, 2)

    """
    assert hurst<0.5, "hurst parameter should be < 0.5"

    dataset = np.zeros((size, n_lags, 2))

    for j in tqdm(range(size), total=size):
        # we generate v process
        m = MBM(n=n_lags-1, hurst=lambda t: hurst, length=maturity, method='riemannliouville')
        fbm = m.mbm() # fractional Brownian motion
        times = m.times()
        V = xi * np.exp(eta * fbm - 0.5 * eta**2 * times**(2*hurst))

        # we generate price process
        h = times[1:] - times[:-1] # time increments
        brownian_increments = np.random.randn(h.shape[0]) * np.sqrt(h)

        log_S = np.zeros_like(V)
        log_S[1:] = (-0.5 * V[:-1]*h + np.sqrt(V[:-1]) * brownian_increments).cumsum() # Ito formula to get SDE for  d log(S_t). We assume S_0 = 1
        S = np.exp(log_S)
        dataset[j] = np.stack([S, V],1) 
    return dataset

def get_gbm(size, n_lags, d=1, drift=0., scale=0.1, h=1):
    x_real = torch.ones(size, n_lags, d)
    x_real[:, 1:, :] = torch.exp(
    (drift - scale ** 2 / 2) * h + (scale * np.sqrt(h) * torch.randn(size, n_lags - 1, d)))
    x_real = x_real.cumprod(1)
    return x_real