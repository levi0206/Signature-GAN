from abc import abstractmethod
import torch
from lib.utils import sample_indices
import signatory

from lib.augmentations import *

class W1_distance():
    '''
    WGAN-GP loss function
    '''
    def __init__(self, D, x_real, x_fake, lambda_reg=100):

        self.dimension = x_real.shape[2] # path dimension
        self.length = x_real.shape[1] # path length

        # Real and fake data
        self.dataset1 = x_real
        self.dataset2 = x_fake

        self.critic = D
        
        # Regularization coefficient
        self.lambda_reg = lambda_reg

    @abstractmethod
    def _input_dim(self):
        ...

    @abstractmethod
    def _get_input(self,x):
        """
        Input given to critic. It will depend on the space where we are calculating the W1-dist (e.g. Path space, or SigSpace)
        Parameters
        ----------
        x: torch.Tensor
            tensor of shape (batch_size, L, d)
        Returns
        -------
        x_out: torch.Tensor
            Tensor of shape (batch_size, d')
        """
        ...

    def get_dist(self, batch_size) -> torch.Tensor:
        device = self.dataset1.device

        # Sample paths
        indices = sample_indices(self.dataset1.shape[0], batch_size, device=device)
        x_fake = self.dataset1[indices] # (batch_size, length, dimension)
        x_real = self.dataset2[indices] # (batch_size, length, dimension)

        # W1 distance: E[D(x_fake)]-E[D(x_real)]
        x_real = self._get_input(x_real) 
        critic_x_real = self.critic(x_real) # (batch_size, probability)
        x_fake = self._get_input(x_fake)
        critic_x_fake = self.critic(x_fake) # (batch_size, probability)
        W1_distance = critic_x_fake.mean() - critic_x_real.mean() 

        # Gradient penalty: lambda * (norm of gradient -1)^2
        eps = torch.rand(batch_size, 1, device=x_real.device)
        x_hat = (1-eps) * x_real + eps * x_fake
        x_hat = x_hat.requires_grad_(True)
        critic_x_hat = self.critic(x_hat) # (batch_size, 1)
        gradient = torch.autograd.grad(critic_x_hat.sum(), x_hat,
                retain_graph=True,
                create_graph=True,
                only_inputs=True)[0]
        gradient_penalty = self.lambda_reg*torch.mean((torch.norm(gradient, p=2, dim=1)-1)**2)
        loss = W1_distance + gradient_penalty
        return loss
    
class W1_dist_PathSpace(W1_distance):
    """
    Class that finds the Wasserstein1 distance between two paths
    """

    def _input_dim(self):
        return self.dimension * self.length
    
    def _get_input(self,x):
        """
        Parameters
        ----------
        x: torch.Tensor
            tensor of shape (batch_size, L, d)
        Returns
        -------
        torch.Tensor of the shape (batch_size, L*d)
        """
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class W1_dist_SigSpace(W1_distance):

    def __init__(self, dataset1, dataset2, lambda_reg=100, depth=3, augmentations=(LeadLag(with_time=True),)):
        
        self.augmentations = augmentations
        self.depth = depth
        super().__init__(dataset1, dataset2,lambda_reg)
    
    def _input_dim(self):
        channels = self.d
        for aug in self.augmentations:
            if isinstance(aug,LeadLag):
                channels = channels*2
            if hasattr(aug, 'with_time'):
                channels = channels+1 if aug.with_time else channels
        return signatory.signature_channels(channels=channels, depth=self.depth)
    
    def _get_input(self, x):
        x_augmented = apply_augmentations(x, self.augmentations)
        return signatory.signature(x_augmented, depth=self.depth, basepoint=True)