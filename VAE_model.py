import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class VAEModel(nn.Module):

    def __init__(self, n_params: int, z_dim: int, h_dim: List[int]):
        super(VAEModel, self).__init__()
        # encoder
        self.e1 = nn.ModuleList([nn.Linear(n_params, h_dim[0], bias=True)])
        self.e1 += [nn.Linear(h_dim[i - 1], h_dim[i], bias=True) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim, bias=True)
        self.e3 = nn.Linear(h_dim[-1], z_dim, bias=True)
        # decoder
        self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1], bias=True)])
        self.d4 += [nn.Linear(h_dim[-i + 1], h_dim[-i], bias=True) for i in range(2, len(h_dim) + 1)]
        self.d5 = nn.Linear(h_dim[0], n_params, bias=True)

    def encoder(self, x):
        h = F.relu(self.e1[0](x))
        for i in range(1, len(self.e1)):
            h = F.relu(self.e1[i](h))
        mean = self.e2(h)
        log_var = self.e3(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(log_var)
        std = torch.exp(log_var).pow(0.5)
        z = mean + std * eps
        return z

    def decoder(self, z):
        params = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            params = F.relu(self.d4[i](params))
        params = self.d5(params)
        return params

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        params = self.decoder(z)
        return params, mean, log_var
