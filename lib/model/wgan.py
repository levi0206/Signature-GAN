import torch
from torch import autograd, nn
from tqdm import tqdm
from collections import defaultdict

from lib.utils import sample_indices
from lib.augmentations import apply_augmentations
from lib.utils import to_numpy

def set_requires_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

class WGAN(nn.Module):
    def __init__(self, D, G, batch_size, epoch, discriminator_steps_per_generator_step, lr_discriminator, test_metrics_test, 
                 lr_generator, x_real: torch.Tensor, lambda_reg=10., **kwargs):
        if kwargs.get('augmentations') is not None:
            self.augmentations = kwargs['augmentations']
            del kwargs['augmentations']
        else:
            self.augmentations = None
        super(WGAN, self).__init__()

        self.D = D
        self.G = G
        self.G_optimizer = torch.optim.Adam(G.parameters(), lr=lr_generator, betas=(0, 0.9))
        self.D_optimizer = torch.optim.Adam(D.parameters(), lr=lr_discriminator, betas=(0, 0.9))
        
        self.batch_size = batch_size
        self.epoch = epoch
        self.discriminator_steps_per_generator_step = discriminator_steps_per_generator_step
        self.lr_discriminator = lr_discriminator
        self.lr_generator = lr_generator
        
        print("augmentations: {}".format(self.augmentations))
        if self.augmentations is not None:
            self.x_real = apply_augmentations(x_real, self.augmentations)
            print("x_real shape: {}".format(self.x_real.shape))
        else:
            self.x_real = x_real

        self.lambda_reg = lambda_reg
        self.losses_history = defaultdict(list)
        self.best_cov_err = None

        self.test_metrics_test = test_metrics_test

        self.device = kwargs['device']
        
    def fit(self, device):
        self.G.to(device)
        self.D.to(device)
        pbar = tqdm(range(self.epoch))
        for i in pbar:
            self.step(device)
            pbar.set_description(
                "G_loss {:1.6e} D_loss {:1.6e}".format(self.losses_history['G_loss'][-1],
                                                                       self.losses_history['D_loss'][-1],))

    def step(self, device):
        
        for i in range(self.discriminator_steps_per_generator_step):
            # Generate x_fake
            indices = sample_indices(self.x_real.shape[0], self.batch_size, self.device)
            x_real_batch = self.x_real[indices].to(device)

            with torch.no_grad():
                x_fake = self.G(batch_size=self.batch_size, window_size=self.x_real.shape[1], device=device)
                if self.augmentations is not None:
                    x_fake = apply_augmentations(x_fake, self.augmentations)

            D_loss = self.D_train(x_fake, x_real_batch)
            if i == 0:
                self.losses_history['D_loss'].append(D_loss)
        G_loss = self.G_train(device)
        self.losses_history['G_loss'].append(G_loss)

    def G_train(self, device):

        set_requires_grad(self.G, True)

        x_fake = self.G(batch_size=self.batch_size, window_size=self.x_real.shape[1], device=device)
        if self.augmentations is not None:
            x_fake = apply_augmentations(x_fake, self.augmentations)

        self.G.train()
        self.G_optimizer.zero_grad()
        D_fake = self.D(x_fake)
        self.D.train()
        G_loss = -D_fake.mean()
        G_loss.backward()
        self.G_optimizer.step()
        self.evaluate(x_fake)

        set_requires_grad(self.G, False)
        return G_loss.item()

    def D_train(self, x_fake, x_real):

        set_requires_grad(self.D, True)

        self.D.train()
        self.D_optimizer.zero_grad()

        x_real.requires_grad_()
        x_fake.requires_grad_()

        D_real = self.D(x_real)
        D_loss_real = D_real.mean()

        x_fake.requires_grad_()
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interpolate = (1 - eps) * x_fake + eps * x_real
        D_fake = self.D(x_interpolate)
        D_loss_fake = D_fake.mean()

        with torch.backends.cudnn.flags(enabled=False):
            # Gradient penalty
            gp = self.lambda_reg * self.gradient_penalty(x_real, x_fake, eps)
            
        total_loss = D_loss_fake - D_loss_real + gp
        total_loss.backward()

        self.D_optimizer.step()

        # Set gradient to False
        set_requires_grad(self.D, False)
        return total_loss.item()
    
    def gradient_penalty(self, x_real, x_fake, eps):
        '''
        "Improved Training of Wasserstein GANs"
        https://arxiv.org/abs/1704.00028        
        '''
        x_interpolate = (1 - eps) * x_fake + eps * x_real
        x_interpolate = x_interpolate.detach()
        x_interpolate.requires_grad_() # W
        prob_interpolate = self.D(x_interpolate)
        
        gradients = autograd.grad(outputs=prob_interpolate.sum(), inputs=x_interpolate,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        assert gradients.shape == x_real.shape
        # print("gradient shape: {}".format(gradients.shape))
        
        gradients = gradients.view(x_fake.shape[0], -1)

        gradients_norm = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradients_norm

    def evaluate(self, x_fake):
        # print("x_fake shape: {}".format(x_fake.shape))
        with torch.no_grad():
            for test_metric in self.test_metrics_test:
                test_metric(x_fake)
                loss = to_numpy(test_metric.loss_componentwise)
                if len(loss.shape) == 1:
                    loss = loss[..., None]
                self.losses_history[test_metric.name + '_test'].append(loss)
        self.best_cov_err = self.losses_history['covariance_test'][-1].item() if self.best_cov_err == None else self.best_cov_err
        if self.losses_history['covariance_test'][-1].item() < self.best_cov_err:
            self.best_cov_err = self.losses_history['covariance_test'][-1].item()