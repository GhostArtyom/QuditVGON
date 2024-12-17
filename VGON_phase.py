import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from scipy.io import savemat
from VAE_model import VAEModel
from Hamiltonian import BBH_model
import torch.distributions as dists
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader
from exact_diagonalization import qutrit_BBH_model
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)
checkpoint = None  # input('Input checkpoint filename: ')

n_layers = 2
n_qudits = 6
n_iter = 2000
batch_size = 16
weight_decay = 1e-2
learning_rate = 1e-3
theta = 0.32 * np.pi

n_qubits = 2 * n_qudits
n_samples = batch_size * n_iter
n_params = n_layers * (n_qudits - 1) * NUM_PR

ham = qutrit_BBH_model(n_qudits, theta, is_csr=True)
ground_state_energy = eigsh(ham, k=1, which='SA', return_eigenvectors=False)[0]

z_dim = 50
list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1)
h_dim = np.power(2, list_z).astype(int)

dev = qml.device('default.qubit', n_qubits)
gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
if torch.cuda.is_available() and gpu_memory < 0.5 and n_qubits >= 12:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

log = f'./logs/VGON_nqd{n_qudits}_phase_202412.log'
logger = Logger(log)
logger.add_handler()

info(f'PyTorch Device: {device}')
info(f'Number of qudits: {n_qudits}')
info(f'Number of qubits: {n_qubits}')
info(f'Weight Decay: {weight_decay:.0e}')
info(f'Learning Rate: {learning_rate:.0e}')
info(f'Ground State Energy: {ground_state_energy:.8f}')


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


Ham = BBH_model(n_qudits, theta)
model = VAEModel(n_params, z_dim, h_dim).to(device)
if checkpoint:
    state_dict = torch.load(f'./mats/{checkpoint}.pt', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    info(f'Load: state_dict of {checkpoint}.pt')
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

    energy_coeff, kl_coeff = 1, 1
    loss = energy_coeff * energy_mean + kl_coeff * kl_div
    loss.backward()
    optimizer.step()

    t = time.perf_counter() - start
    energy_str = f'Energy: {energy.max():.8f}, {energy.mean():.8f}, {energy.min():.8f}'
    info(f'Loss: {loss:.8f}, {energy_str}, KL: {kl_div:.4e}, {i+1}/{n_iter}, {t:.2f}')

    energy_gap = energy_mean - ground_state_energy
    energy_tol, kl_tol = 1e-2, 1e-5
    if (i + 1) % 500 == 0 or i + 1 >= n_iter:
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        path = f'./mats/VGON_nqd{n_qudits}_{time_str}'
        mat_dict = {
            'theta': theta,
            'n_iter': n_iter,
            'loss': loss.item(),
            'n_qudits': n_qudits,
            'n_qubits': n_qubits,
            'kl_div': kl_div.item(),
            'batch_size': batch_size,
            'energy': energy_mean.item(),
            'n_train': f'{i+1}/{n_iter}',
            'weight_decay': weight_decay,
            'learning_rate': learning_rate
        }
        savemat(f'{path}.mat', mat_dict)
        torch.save(model.state_dict(), f'{path}.pt')
        info(f'Energy Gap: {energy_gap:.4e}, KL: {kl_div:.4e}, {i+1}/{n_iter}, Save: {path}.mat&pt')
