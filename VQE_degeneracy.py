import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from typing import List
from logging import info
from logger import Logger
from utils import updatemat
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

n_layers = 2
n_qudits = 4
epochs = 1000
beta = -1 / 3
repetition = 3
learning_rate = 1e-3
n_qubits = 2 * n_qudits

dev = qml.device('default.qubit', n_qubits)
gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
if torch.cuda.is_available() and gpu_memory < 0.5 and n_qubits >= 14:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

log = f'./logs/VQE_nqd{n_qudits}_degeneracy.log'
logger = Logger(log)
logger.add_handler()


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
    params = params.reshape(n_layers, n_qudits - 1, NUM_PR)
    qml.layer(qutrit_symmetric_ansatz, n_layers, params)
    return qml.state()


@qml.qnode(dev, interface='torch', diff_method='best')
def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
    params = params.reshape(n_layers, n_qudits - 1, NUM_PR)
    qml.layer(qutrit_symmetric_ansatz, n_layers, params)
    return qml.expval(Ham)


def running(n_layers: int, n_qudits: int, beta: float, epochs: int, learning_rate: float):
    logger.add_handler()
    info(f'Repeat: {r+1}, Learning Rate: {learning_rate}')
    info(f'Coefficient beta: {beta:.4f}')
    info(f'Number of qudits: {n_qudits}')
    info(f'Number of qubits: {n_qubits}')
    info(f'PyTorch Device: {device}')

    n_params = n_layers * (n_qudits - 1) * NUM_PR
    params_init = np.random.uniform(-np.pi, np.pi, n_params)
    params = torch.tensor(params_init, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=learning_rate)
    Ham = Hamiltonian(n_qudits, beta)

    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = circuit_expval(n_layers, params, Ham)
        loss.backward()
        optimizer.step()
        count = epoch + 1
        if count % 10 == 0:
            t = time.perf_counter() - start
            info(f'Loss: {loss.item():.20f}, {count}/{epochs}, {t:.2f}')

    params_res = params.detach().cpu().numpy()
    state_res = circuit_state(n_layers, params_res)
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mat_dict = {
        f'T{time_str}': {
            'n_qudits': n_qudits,
            'n_qubits': n_qubits,
            'beta': beta,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'params_init': params_init,
            'params_res': params_res,
            'state_res': state_res,
            'loss_res': loss.item()
        }
    }
    mat_path = f'./mats/VQE_degeneracy.mat'
    updatemat(mat_path, mat_dict)
    info(f'Save: {mat_path} T{time_str}')
    logger.remove_handler()


for r in range(repetition):
    running(n_layers, n_qudits, beta, epochs, learning_rate)
