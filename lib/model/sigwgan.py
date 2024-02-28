import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict

from lib.distance.sigw1metric import SigW1Metric
from lib.utils import to_numpy

class SigWGAN(nn.Module): 
    def __init__(self, D, G, lr_generator, lr_discriminator, epoch, batch_size, depth, x_real_rolled, augmentations, 
                test_metrics_test, normalise_sig: bool = True, mask_rate=0.01,
                 **kwargs):
        super(SigWGAN, self).__init__()
        self.sig_w1_metric = SigW1Metric(depth=depth, x_real=x_real_rolled, augmentations=augmentations,
                                         mask_rate=mask_rate, normalise=normalise_sig)
        self.D = D
        self.G = G

        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.epoch = epoch
        self.batch_size = batch_size
        self.depth = depth
        self.augmentations = augmentations

        self.test_metrics_test = test_metrics_test

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr_generator)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr_discriminator, betas=(0, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=0.95, step_size=128)

        self.losses_history = defaultdict(list)

    def fit(self, device):
        self.G.to(device)
        best_loss = 10**10
        pbar = tqdm(range(self.epoch))
        for j in pbar:
            self.G_optimizer.zero_grad()
            x_fake = self.G(
                batch_size=self.batch_size, window_size=self.sig_w1_metric.window_size, device=device
            )
            loss = self.sig_w1_metric(x_fake)  # E[S(x_real)] - E[S(X_fake)]
            loss.backward()
            best_loss = loss.item() if j == 0 else best_loss

            pbar.set_description("sig-w1 loss: {:.4f}".format(loss.item()))
            self.G_optimizer.step()
            self.scheduler.step()
            self.losses_history['sig_w1_loss'].append(loss.item())
            self.evaluate(x_fake)

    def evaluate(self, x_fake):
        with torch.no_grad():
            for test_metric in self.test_metrics_test:
                test_metric(x_fake)
                loss = to_numpy(test_metric.loss_componentwise)
                if len(loss.shape) == 1:
                    loss = loss[..., None]
                self.losses_history[test_metric.name + '_test'].append(loss)