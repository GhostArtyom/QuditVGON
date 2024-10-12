# Are eigvals of origin Hamiltonian same as encoded Hamiltonian?

import sys
import time
import torch
import logging
import numpy as np
import torch.nn as nn
import pennylane as qml
from typing import List
from logging import info
from scipy.io import savemat
import torch.nn.functional as F
from itertools import combinations
import torch.distributions as dists
from torch.utils.data import DataLoader
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)

n_layers = 2
n_qudits = 7
beta = -1 / 3
n_iter = 5000
batch_size = 10
learning_rate = 1e-3
energy_coeff, kl_coeff = 1, 1

n_qubits = 2 * n_qudits
n_samples = batch_size * n_iter
n_params = n_layers * (n_qudits - 1) * NUM_PR
ground_state_energy = -2 / 3 * (n_qudits - 1)

z_dim = 50
list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1)
h_dim = np.power(2, list_z).astype(int)

dev = qml.device('default.qubit', n_qubits)
if torch.cuda.is_available() and n_qubits >= 14:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log = f'./logs/VGON_nqd{n_qudits}.log'
file_handler = logging.FileHandler(log)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

info(f'PyTorch Device: {device}')
info(f'Number of qudits: {n_qudits}')
info(f'Number of qubits: {n_qubits}')
info(f'Learning Rate: {learning_rate:.0e}')
info(f'Ground state energy: {ground_state_energy:.4f}')


def spin_operator(obj: List[int]):
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    sx = qml.X(obj[0]) + qml.X(obj[1])
    sy = qml.Y(obj[0]) + qml.Y(obj[1])
    sz = qml.Z(obj[0]) + qml.Z(obj[1])
    return sx, sy, sz


def spin_operator2(obj: List[int]):
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    s1 = spin_operator(obj)
    s2 = [i @ j for i in s1 for j in s1]
    return s2


def Hamiltonian(n_qudits: int, beta: float):
    ham1, ham2 = 0, 0
    for i in range(n_qudits - 1):
        obj1 = [2 * i, 2 * i + 1]
        obj2 = [2 * i + 2, 2 * i + 3]
        ham1 += qml.sum(*[spin_operator(obj1)[i] @ spin_operator(obj2)[i] for i in range(3)])
        ham2 += qml.sum(*[spin_operator2(obj1)[i] @ spin_operator2(obj2)[i] for i in range(9)])
    ham = ham1 / 4 - beta * ham2 / 16
    coeffs, obs = qml.simplify(ham).terms()
    coeffs = torch.tensor(coeffs).real
    return qml.Hamiltonian(coeffs, obs)


def qutrit_symmetric_ansatz(params: torch.Tensor):
    for i in range(n_qudits - 1):
        obj = list(range(n_qubits - 2 * i - 4, n_qubits - 2 * i))
        two_qutrit_unitary_synthesis(params[i], obj)


@qml.qnode(dev, interface='torch', diff_method='best')
def circuit_state(n_layers: int, params: torch.Tensor):
    params = params.reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
    qml.layer(qutrit_symmetric_ansatz, n_layers, params)
    return qml.state()


@qml.qnode(dev, interface='torch', diff_method='best')
def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
    params = params.reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
    qml.layer(qutrit_symmetric_ansatz, n_layers, params)
    return qml.expval(Ham)


class VAE_Model(nn.Module):

    def __init__(self, n_params: int, z_dim: int, h_dim: List[int]):
        super(VAE_Model, self).__init__()
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
        eps = torch.randn(log_var.shape).to(device)
        std = torch.exp(log_var).pow(0.5)
        z = mean + std * eps
        return z

    def decoder_expval(self, z):
        params = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            params = F.relu(self.d4[i](params))
        params = self.d5(params)
        return circuit_expval(n_layers, params, Ham)

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


Ham = Hamiltonian(n_qudits, beta)

model = VAE_Model(n_params, z_dim, h_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_dist = dists.Uniform(0, 1).sample([n_samples, n_params])
train_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)

start = time.perf_counter()
for i, batch in enumerate(train_data):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    params, mean, log_var = model(batch.to(device))

    energy = circuit_expval(n_layers, params, Ham)
    energy = energy.mean()

    kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kl_div = kl_div.mean()

    cos_sims = torch.empty((0), device=device)
    for ind in combinations(range(batch_size), 2):
        sim = torch.cosine_similarity(params[ind[0], :], params[ind[1], :], dim=0)
        cos_sims = torch.cat((cos_sims, sim.unsqueeze(0)), dim=0)
    cos_sim = cos_sims.mean()

    cos_sim_coeff = 9 * (2 * cos_sim - cos_sim.pow(2)) + 1
    loss = energy_coeff * energy + kl_coeff * kl_div + cos_sim_coeff * cos_sim
    loss.backward()
    optimizer.step()

    t = time.perf_counter() - start
    loss, energy, kl_div, cos_sim = loss.item(), energy.item(), kl_div.item(), cos_sim.item()
    info(f'Loss: {loss:.8f}, Energy: {energy:.8f}, KL: {kl_div:.4e}, Cos_Sim: {cos_sim:.8f} * {cos_sim_coeff:.2f}, {i+1}/{n_iter}, {t:.2f}')

    energy_tol, kl_tol = 1e-2, 1e-5
    energy_gap = energy - ground_state_energy
    if energy_gap < energy_tol and kl_div < kl_tol or i >= n_iter - 1:
        params_res = params.detach().cpu().numpy()
        state_res = circuit_state(n_layers, params_res)
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        path = f'./mats/VGON_nqd{n_qudits}_{time_str}'
        mat_dict = {
            'beta': beta,
            'loss': loss,
            'n_iter': n_iter,
            'energy': energy,
            'kl_div': kl_div,
            'cos_sim': cos_sim,
            'state': state_res,
            'params': params_res,
            'n_qudits': n_qudits,
            'n_qubits': n_qubits,
            'batch_size': batch_size,
            'energy_tol': energy_tol,
            'learning_rate': learning_rate
        }
        savemat(f'{path}.mat', mat_dict)
        torch.save(model.state_dict(), f'{path}.pt')
        info(f'Energy Gap: {energy_gap:.4e}, KL: {kl_div:.4e}, Save: {path}.mat&pt')
