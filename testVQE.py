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
    sx = qml.X(obj[0]) / 2 + qml.X(obj[1]) / 2
    sy = qml.Y(obj[0]) / 2 + qml.Y(obj[1]) / 2
    sz = qml.Z(obj[0]) / 2 + qml.Z(obj[1]) / 2
    return sx + sy + sz


def spin_operator2(obj: List[int]):
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    s1 = qml.X(obj[0]) + qml.Y(obj[0]) + qml.Z(obj[0])
    s2 = qml.X(obj[1]) + qml.Y(obj[1]) + qml.Z(obj[1])
    return 3 / 2 * qml.I(obj) + (s1 @ s2) / 2


def Hamiltonian(n_qudits: int, beta: float):
    Ham = 0
    for i in range(n_qudits - 1):
        obj1 = [2 * i, 2 * i + 1]
        obj2 = [2 * i + 2, 2 * i + 3]
        Ham += spin_operator(obj1) @ spin_operator(obj2)
        Ham -= beta * (spin_operator2(obj1) @ spin_operator2(obj2))
    return qml.simplify(Ham)


def qutrit_symmetric_ansatz(n_qudits: int, params: torch.Tensor, Ham):
    params = params.reshape(n_qudits - 1, NUM_PR)
    n_qubits = 2 * n_qudits
    for i in range(n_qudits - 1):
        obj = list(range(n_qubits - 2 * i - 4, n_qubits - 2 * i))
        two_qutrit_unitary_synthesis(params[i], obj)
    return qml.expval(Ham)


def running(n_qudits: int, beta: float, epochs: int):
    n_qubits = 2 * n_qudits
    dev = qml.device('default.qubit', n_qubits)
    info(f'Number of qudits: {n_qudits}')
    info(f'Number of qubits: {n_qubits}')
    info(f'Number of epochs: {epochs}')

    if torch.cuda.is_available() and n_qubits > 14:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    info(f'PyTorch Device: {device}')

    pr_num = (n_qudits - 1) * NUM_PR
    init_params = np.random.uniform(-np.pi, np.pi, pr_num)
    params = torch.tensor(init_params, device=device, requires_grad=True)
    cost_fn = qml.QNode(qutrit_symmetric_ansatz, dev, interface='torch')
    optimizer = torch.optim.Adam([params], lr=0.1)
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
    nqd_time = f'nqd{n_qudits}_{epochs}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}'
    mat_dict = {f'loss_{nqd_time}': loss_res, f'pr_{nqd_time}': params_res}
    updatemat('./mats/testVQE.mat', mat_dict)


# n_qudits, beta, epochs = 4, -0.3, 500
n_qudits, beta = 4, -0.3
for epochs in [500, 1000, 1500, 2000]:
    running(n_qudits, beta, epochs)
