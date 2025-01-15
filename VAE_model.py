import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class VAEModel(nn.Module):

    def __init__(self, n_params: int, h_dim: List[int], z_dim: int):
        super(VAEModel, self).__init__()
        self.encoder = nn.ModuleList([nn.Linear(n_params, h_dim[0], bias=True)])
        self.encoder += [nn.Linear(h_dim[i - 1], h_dim[i], bias=True) for i in range(1, len(h_dim))]
        self.encoder_mean = nn.Linear(h_dim[-1], z_dim, bias=True)
        self.encoder_log_var = nn.Linear(h_dim[-1], z_dim, bias=True)
        self.decoder = nn.ModuleList([nn.Linear(z_dim, h_dim[-1], bias=True)])
        self.decoder += [nn.Linear(h_dim[1 - i], h_dim[-i], bias=True) for i in range(2, len(h_dim) + 1)]
        self.decoder += [nn.Linear(h_dim[0], n_params, bias=True)]

    def encode(self, x):
        h = F.relu(self.encoder[0](x))
        for i in range(1, len(self.encoder)):
            h = F.relu(self.encoder[i](h))
        mean = self.encoder_mean(h)
        log_var = self.encoder_log_var(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(log_var)
        std = torch.exp(log_var).sqrt()
        z = mean + eps * std
        return z

    def decode(self, z):
        h = F.relu(self.decoder[0](z))
        for i in range(1, len(self.decoder) - 1):
            h = F.relu(self.decoder[i](h))
        params = self.decoder[-1](h)
        return params

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        params = self.decode(z)
        return params, mean, log_var
