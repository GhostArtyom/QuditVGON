import time
import torch
import GPUtil
import numpy as np
import torch.nn as nn
import pennylane as qml
from typing import List
from logging import info
from logger import Logger
from scipy.io import savemat
import torch.nn.functional as F
from itertools import combinations
import torch.distributions as dists
from torch.utils.data import DataLoader
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)
checkpoint = None  # input('Input checkpoint filename: ')

n_layers = 2
n_qudits = 7
beta = -1 / 3
n_iter = 2000
batch_size = 16
weight_decay = 1e-2
learning_rate = 1e-3
energy_coeff, kl_coeff = 1, 1
energy_tol, kl_tol = 1e-2, 1e-5

n_qubits = 2 * n_qudits
n_samples = batch_size * n_iter
n_params = n_layers * (n_qudits - 1) * NUM_PR
ground_state_energy = -2 / 3 * (n_qudits - 1)

z_dim = 50
list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1)
h_dim = np.power(2, list_z).astype(int)

dev = qml.device('default.qubit', n_qubits)
gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
if torch.cuda.is_available() and gpu_memory < 0.5 and n_qubits >= 14:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

log = f'./logs/VGON_nqd{n_qudits}_degeneracy.log'
logger = Logger(log)
logger.add_handler()

info(f'PyTorch Device: {device}')
info(f'Number of qudits: {n_qudits}')
info(f'Number of qubits: {n_qubits}')
info(f'Weight Decay: {weight_decay:.0e}')
info(f'Learning Rate: {learning_rate:.0e}')
info(f'Ground State Energy: {ground_state_energy:.4f}')


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
    params = params.transpose(0, 1).reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
    qml.layer(qutrit_symmetric_ansatz, n_layers, params)
    return qml.state()


@qml.qnode(dev, interface='torch', diff_method='best')
def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
    params = params.transpose(0, 1).reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
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
if checkpoint:
    state_dict = torch.load(f'./mats/{checkpoint}.pt', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

data_dist = dists.Uniform(0, 1).sample([n_samples, n_params])
train_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)

start = time.perf_counter()
for i, batch in enumerate(train_data):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    params, mean, log_var = model(batch.to(device))

    energy = circuit_expval(n_layers, params, Ham)
    energy_mean = energy.mean()

    kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kl_div = kl_div.mean()

    cos_sims = torch.empty((0), device=device)
    for ind in combinations(range(batch_size), 2):
        cos_sim = torch.cosine_similarity(params[ind[0], :], params[ind[1], :], dim=0)
        cos_sims = torch.cat((cos_sims, cos_sim.unsqueeze(0)), dim=0)
    cos_sim_max = cos_sims.max()
    cos_sim_mean = cos_sims.mean()

    coeff = (energy_mean - ground_state_energy).ceil()
    cos_sim_max_coeff = (75 * (cos_sim_max - 0.8)).ceil() if cos_sim_max > 0.8 else 0
    cos_sim_mean_coeff = coeff / 10 * (10 * cos_sim_mean).ceil() if cos_sim_mean > 0 else 0
    loss = energy_coeff * energy_mean + kl_coeff * kl_div + cos_sim_max_coeff * cos_sim_max + cos_sim_mean_coeff * cos_sim_mean
    loss.backward()
    optimizer.step()

    t = time.perf_counter() - start
    cos_sim_str = f'Cos_Sim: {cos_sim_max_coeff:.0f}*{cos_sim_max:.8f}, {cos_sim_mean_coeff:.1f}*{cos_sim_mean:.8f}, {cos_sims.min():.8f}'
    info(f'Loss: {loss:.8f}, Energy: {energy_mean:.8f}, KL: {kl_div:.4e}, {cos_sim_str}, {i+1}/{n_iter}, {t:.2f}')

    energy_gap = energy_mean - ground_state_energy
    if (i + 1) % 500 == 0 or i + 1 >= n_iter or (energy_gap < energy_tol and kl_div < kl_tol):
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        path = f'./mats/VGON_nqd{n_qudits}_{time_str}'
        mat_dict = {
            'beta': beta,
            'n_iter': n_iter,
            'loss': loss.item(),
            'n_qudits': n_qudits,
            'n_qubits': n_qubits,
            'kl_div': kl_div.item(),
            'batch_size': batch_size,
            'energy_tol': energy_tol,
            'energy': energy_mean.item(),
            'n_train': f'{i+1}/{n_iter}',
            'weight_decay': weight_decay,
            'learning_rate': learning_rate,
            'cos_sim_max': cos_sim_max.item(),
            'cos_sim_mean': cos_sim_mean.item()
        }
        savemat(f'{path}.mat', mat_dict)
        torch.save(model.state_dict(), f'{path}.pt')
        info(f'Energy Gap: {energy_gap:.4e}, KL: {kl_div:.4e}, {i+1}/{n_iter}, Save: {path}.mat&pt')
