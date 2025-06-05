import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from scipy.io import savemat
from datetime import datetime
from Hamiltonian import AKLT_model
import torch.distributions as dists
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

n_layers = 2
n_qudits = 6
n_iter = 2000
beta = -1 / 3
batch_size = 16
weight_decay = 1e-2
learning_rate = 1e-3

n_qubits = 2 * n_qudits
n_params = n_layers * (n_qudits - 1) * NUM_PR
ground_state_energy = -2 / 3 * (n_qudits - 1)

dev = qml.device('default.qubit', n_qubits)
gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
if torch.cuda.is_available() and gpu_memory < 0.8 and n_qubits >= 12:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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


year_month = datetime.today().strftime('%Y%m')
log = f'./logs/VQE_nqd{n_qudits}_degeneracy_{year_month}.log'
logger = Logger(log)
logger.add_handler()
info(f'PyTorch Device: {device}')
info(f'Number of qudits: {n_qudits}')
info(f'Number of qubits: {n_qubits}')
info(f'Weight Decay: {weight_decay:.0e}')
info(f'Learning Rate: {learning_rate:.0e}')


def running(n_iter: int):
    Ham = AKLT_model(n_qudits, beta)
    params = dists.Uniform(0, 2 * np.pi).sample([batch_size, n_params]).to(device)
    params.requires_grad_(True)
    optimizer = torch.optim.AdamW([params], lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    for i in range(n_iter):
        optimizer.zero_grad(set_to_none=True)
        energy = circuit_expval(n_layers, params, Ham)
        loss = energy.mean()
        loss.backward()
        optimizer.step()
        t = time.perf_counter() - start
        energy_gap = loss - ground_state_energy
        info(f'Loss: {loss:.8f}, {energy_gap:.4e}, {i+1}/{n_iter}, {r+1}/{repeat}, {t:.2f}')

    params_res = params.detach().cpu().numpy()
    state_res = circuit_state(n_layers, params_res)
    mat_dict = {
        'beta': beta,
        'n_iter': n_iter,
        'n_qudits': n_qudits,
        'n_qubits': n_qubits,
        'loss_res': loss.item(),
        'state_res': state_res,
        'params_res': params_res,
        'weight_decay': weight_decay,
        'learning_rate': learning_rate
    }
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    path = f'./mats/VQE_nqd{n_qudits}_{time_str}.mat'
    savemat(path, mat_dict)
    info(f'Save: {path}')


repeat = 3
for r in range(repeat):
    logger.add_handler()
    running(n_iter)
    logger.remove_handler()
