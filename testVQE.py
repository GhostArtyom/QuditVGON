import time
import torch
import numpy as np
import pennylane as qml
from typing import List
from utils import updatemat
from logging import info, INFO, basicConfig
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

basicConfig(filename='./logs/testVQE.log', format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)


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
        for j in range(3):
            ham1 += spin_operator(obj1)[j] @ spin_operator(obj2)[j]
        for k in range(9):
            ham2 += spin_operator2(obj1)[k] @ spin_operator2(obj2)[k]
    Ham = ham1 / 4 - beta * ham2 / 16
    return Ham


def qutrit_symmetric_ansatz(n_qudits: int, params: torch.Tensor, Ham):
    params = params.reshape(n_qudits - 1, NUM_PR)
    n_qubits = 2 * n_qudits
    for i in range(n_qudits - 1):
        obj = list(range(n_qubits - 2 * i - 4, n_qubits - 2 * i))
        two_qutrit_unitary_synthesis(params[i], obj)
    return qml.expval(Ham)


def running(n_qudits: int, beta: float, epochs: int, lr: float):
    n_qubits = 2 * n_qudits
    dev = qml.device('default.qubit', n_qubits)
    info(f'Coefficient beta: {beta}')
    info(f'Number of qudits: {n_qudits}')
    info(f'Number of qubits: {n_qubits}')

    if torch.cuda.is_available() and n_qubits > 14:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    info(f'PyTorch Device: {device}')

    pr_num = (n_qudits - 1) * NUM_PR
    init_params = np.random.uniform(-np.pi, np.pi, pr_num)
    params = torch.tensor(init_params, device=device, requires_grad=True)
    cost_fn = qml.QNode(qutrit_symmetric_ansatz, dev, interface='torch')
    optimizer = torch.optim.Adam([params], lr=lr)
    Ham = Hamiltonian(n_qudits, beta)

    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = cost_fn(n_qudits, params, Ham)
        loss.backward()
        optimizer.step()
        count = epoch + 1
        if count % 10 == 0:
            t = time.perf_counter() - start
            print(f'Loss: {loss.item():.20f}, {count}/{epochs}, {t:.2f}')
            info(f'Loss: {loss.item():.20f}, {count}/{epochs}, {t:.2f}')

    loss_res = loss.detach().cpu()
    params_res = optimizer.param_groups[0]['params'][0].detach().cpu()
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mat_dict = {
        f'T{time_str}': {
            'n_qudits': n_qudits,
            'n_qubits': n_qubits,
            'beta': beta,
            'epochs': epochs,
            'learning_rate': lr,
            'params_init': init_params,
            'params_res': params_res,
            'loss': loss_res
        }
    }
    updatemat(f'./mats/testVQE.mat', mat_dict)


n_qudits, beta, epochs = 4, -0.3, 1000
for r in range(2):
    for lr in [1e-3, 5e-3, 1e-4]:
        info(f'Repeat: {r+1}, Learning Rate: {lr}')
        running(n_qudits, beta, epochs, lr)
