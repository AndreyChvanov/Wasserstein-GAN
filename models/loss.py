import torch
import torch.nn as nn
from torch import autograd


class DiscriminatorLoss(nn.Module):
    def __init__(self, config, device):
        super(DiscriminatorLoss, self).__init__()
        self.device = device
        self.penalty_lambda = config.penalty_lambda
        self.epsilon = torch.FloatTensor(config.batch_size, 1, 1, 1).to(device)

    def forward(self, netD, fake, disc_fake, real, disc_real):
        loss = disc_fake - disc_real
        self.epsilon.uniform_(0, 1)
        interpolates = self.epsilon * real + (1 - self.epsilon) * fake
        disc_interpolates = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device), create_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * self.penalty_lambda
        loss += gradient_penalty
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake):
        
        return -fake

