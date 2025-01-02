import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from utils import fidelity
from scipy.linalg import orth
from Hamiltonian import BBH_model
import torch.distributions as dists
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import eigsh
from qudit_mapping import symmetric_decoding
from exact_diagonalization import qutrit_BBH_model
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis


def running(n_layers: int, n_qudits: int, n_iter: int, batch_size: int, theta: float, checkpoint: str = None):
    weight_decay = 1e-2
    learning_rate = 1e-2
    n_qubits = 2 * n_qudits
    n_params = n_layers * (n_qudits - 1) * NUM_PR
    phase = 'arctan(1/3)' if theta == np.arctan(1 / 3) else f'{theta/np.pi:.2f}π'

    dev = qml.device('default.qubit', n_qubits)
    gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
    if torch.cuda.is_available() and gpu_memory < 0.8 and n_qubits >= 12:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    log = f'./logs/VQE_nqd{n_qudits}_L{n_layers}_phase_202501.log'
    logger = Logger(log)
    logger.add_handler()

    info(f'PyTorch Device: {device}')
    info(f'Number of layers: {n_layers}')
    info(f'Number of qudits: {n_qudits}')
    info(f'Number of qubits: {n_qubits}')
    info(f'Weight Decay: {weight_decay:.0e}')
    info(f'Learning Rate: {learning_rate:.0e}')
    info(f'Coefficient phase: {phase}')

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

    qubit_Ham = BBH_model(n_qudits, theta)
    qutrit_Ham = qutrit_BBH_model(n_qudits, theta, is_csr=True)

    ground_state_energy, ground_states = eigsh(qutrit_Ham, k=4, which='SA')
    ind = np.where(np.isclose(ground_state_energy, ground_state_energy.min()))
    ground_state_energy = ground_state_energy.min()
    info(f'Ground State Energy: {ground_state_energy:.8f}')

    ground_states = ground_states[:, ind[0]]
    ground_states = orth(ground_states).T
    ground_states[np.abs(ground_states) < 1e-15] = 0
    degeneracy = ground_states.shape[0]
    info(f'Degree of degeneracy: {degeneracy}')

    if checkpoint:
        params = torch.from_numpy(load['params_res']).to(device)
        learning_rate = 1e-3
        info(f'Load: params of {checkpoint}.mat, Learning Rate: 1e-3')
    else:
        params = dists.Uniform(0, 2 * np.pi).sample([batch_size, n_params]).to(device)
    params.requires_grad_(True)
    optimizer = torch.optim.AdamW([params], lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    energy_iter = np.empty((0, batch_size))
    fidelity_iter = np.empty((0, batch_size, degeneracy))
    for i in range(n_iter):
        optimizer.zero_grad(set_to_none=True)
        energy = circuit_expval(n_layers, params, qubit_Ham)
        energy_mean = energy.mean()
        energy_mean.backward()
        optimizer.step()

        energy = energy.detach().cpu().numpy()
        energy_iter = np.vstack((energy_iter, energy))
        energy_gap = energy.mean() - ground_state_energy

        states = circuit_state(n_layers, params).detach().cpu().numpy()
        fidelities = np.empty((0, degeneracy))
        for state in states:
            decoded_state = symmetric_decoding(state, n_qudits)
            overlap = [fidelity(decoded_state, ground_state) for ground_state in ground_states]
            fidelities = np.vstack((fidelities, overlap))
        fidelity_iter = np.concatenate((fidelity_iter, fidelities[np.newaxis]))
        fidelity_mean = fidelities.mean(axis=0)
        fidelity_sum = fidelity_mean.sum()
        fidelity_gap = 1 - fidelity_sum

        t = time.perf_counter() - start
        info(f'Energy: {energy_mean:.8f}, {energy_gap:.4e}, Fidelity: {fidelity_sum:.8f}, {fidelity_gap:.4e}, {i+1}/{n_iter}, {t:.2f}')

    params = params.detach().cpu().numpy()
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    path = f'./mats/VQE_nqd{n_qudits}_L{n_layers}_{time_str}.mat'
    mat_dict = {
        'theta': theta,
        'phase': phase,
        'n_iter': n_iter,
        'loss': energy_mean.item(),
        'n_layers': n_layers,
        'n_qudits': n_qudits,
        'n_qubits': n_qubits,
        'states_res': states,
        'params_res': params,
        'batch_size': batch_size,
        'fidelity': fidelity_mean,
        'n_train': f'{i+1}/{n_iter}',
        'weight_decay': weight_decay,
        'learning_rate': learning_rate,
        'ground_states': ground_states,
        'energy': energy.mean().item(),
        'energy_iter': energy_iter.squeeze(),
        'fidelity_iter': fidelity_iter.squeeze(),
        'ground_state_energy': ground_state_energy
    }
    savemat(path, mat_dict)
    info(f'Save: {path}, {i+1}/{n_iter}')
    torch.cuda.empty_cache()
    logger.remove_handler()


n_qudits = 7
n_iter = 500
batch_size = 1

# coeffs = np.array([-0.74, -0.26, -0.24, 0.24, 0.26, 0.49]) * np.pi
coeffs = np.linspace(-0.70, 0.45, 24) * np.pi

checkpoint = None
if checkpoint:
    load = loadmat(f'./mats/{checkpoint}.mat')
    coeffs = [load['theta'].item()]
    n_layers = load['n_layers'].item()
    n_qudits = load['n_qudits'].item()
    batch_size = load['batch_size'].item()

n_layers = int(input('Input layers: '))
for theta in coeffs:
    running(n_layers, n_qudits, n_iter, batch_size, theta, checkpoint)
