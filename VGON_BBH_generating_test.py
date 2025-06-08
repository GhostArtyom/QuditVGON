import os
import re
import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from scipy.io import loadmat
from scipy.linalg import orth
from datetime import datetime
from VAE_model import VAEModel
from Hamiltonian import BBH_model
import torch.distributions as dists
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader
from qudit_mapping import symmetric_decoding
from exact_diagonalization import qutrit_BBH_model
from utils import fidelity, updatemat, extract_datetime, compare_datetime
from qutrit_synthesis import single_qutrit_unitary_synthesis, controlled_diagonal_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)


def generating(n_layers: int, n_qudits: int, n_test: int, batch_size: int, theta: float, phase: str, path: str):
    n_qubits = 2 * n_qudits
    n_samples = batch_size * n_test
    n_params_per_layer = 12 * n_qudits - 3
    n_params = n_layers * n_params_per_layer

    z_dim = 50
    list_z = np.arange(*np.floor(np.log2([n_params, z_dim])), -1)
    h_dim = np.power(2, list_z).astype(int)

    dev = qml.device('default.qubit', n_qubits)
    gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
    if torch.cuda.is_available() and gpu_memory < 0.8 and n_qubits >= 12:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    year_month = datetime.today().strftime('%Y%m')
    log = f'./logs/VGON_nqd{n_qudits}_generating_{year_month}.log'
    logger = Logger(log)
    logger.add_handler()
    info(f'Load: {path}, {n_train}')
    info(f'PyTorch Device: {device}')
    info(f'Number of layers: {n_layers}')
    info(f'Number of qudits: {n_qudits}')
    info(f'Number of qubits: {n_qubits}')
    info(f'Number of params: {n_params}')
    info(f'Size of Encoder: {h_dim}')
    info(f'Coefficient phase: {phase}')

    def qutrit_symmetric_ansatz(params: torch.Tensor):
        for i in range(n_qudits):
            obj = [2 * i, 2 * i + 1]
            single_qutrit_unitary_synthesis(params[9 * i:9 * i + 9], obj)
        for i in range(n_qudits - 1):
            j = 9 * n_qudits + 3 * i
            obj = list(range(2 * i, 2 * i + 4))
            controlled_diagonal_synthesis(params[j:j + 3], 2, obj[-1], obj[::-1][1:])

    @qml.qnode(dev, interface='torch', diff_method='best')
    def circuit_state(n_layers: int, params: torch.Tensor):
        params = params.transpose(0, 1).reshape(n_layers, n_params_per_layer, batch_size)
        qml.layer(qutrit_symmetric_ansatz, n_layers, params)
        return qml.state()

    @qml.qnode(dev, interface='torch', diff_method='best')
    def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
        params = params.transpose(0, 1).reshape(n_layers, n_params_per_layer, batch_size)
        qml.layer(qutrit_symmetric_ansatz, n_layers, params)
        return qml.expval(Ham)

    qubit_Ham = BBH_model(n_qudits, theta)
    qutrit_Ham = qutrit_BBH_model(n_qudits, theta)

    ground_state_energy, ground_states = eigsh(qutrit_Ham, k=6, which='SA')
    ind = np.where(np.isclose(ground_state_energy, ground_state_energy.min()))
    ground_state_energy = ground_state_energy.min()
    energy_gap = load['energy'].item() - ground_state_energy
    energy_gap_tol = np.ceil(energy_gap * 10) / 10
    info(f'Ground State Energy: {ground_state_energy:.8f}, Gap Tolerance: {energy_gap_tol}')

    ground_states = ground_states[:, ind[0]]
    ground_states = orth(ground_states).T
    ground_states[np.abs(ground_states) < 1e-15] = 0
    degeneracy = ground_states.shape[0]
    overlaps = np.empty((0, degeneracy))
    if degeneracy < 4:
        raise ValueError(f'Wrong degeneracy {degeneracy} < 4')

    state_dict = torch.load(f'{path[:-4]}.pt', map_location=device, weights_only=True)
    model = VAEModel(n_params, h_dim, z_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_dist = dists.Normal(0, 1).sample([n_samples, n_params])
    test_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)

    count, count_str = 0, ''
    start = time.perf_counter()
    for i, batch in enumerate(test_data):
        with torch.no_grad():
            params, _, _ = model(batch.to(device))
        energy = circuit_expval(n_layers, params, qubit_Ham)
        energy_str = f'Energy: {energy.max():.8f}, {energy.mean():.8f}, {energy.min():.8f}, Gap: {energy.mean()-ground_state_energy:.4e}'
        energy_ind = torch.where(energy < ground_state_energy + energy_gap_tol)[0]
        count += len(energy_ind)
        count_str = f'{count}/{(i+1)*batch_size}'

        states = circuit_state(n_layers, params)
        states = states[energy_ind].detach().cpu().numpy()
        for state in states:
            decoded_state = symmetric_decoding(state, n_qudits)
            overlap = [fidelity(decoded_state, ground_state) for ground_state in ground_states]
            overlaps = np.vstack((overlaps, overlap))

        t = time.perf_counter() - start
        info(f'{energy_str}, {count_str}, {i+1}/{n_test}, {t:.2f}')

    updatemat(path, {'count': count_str, 'overlaps': overlaps})
    info(f'Save: {path} with count and overlaps')
    logger.remove_handler()


n_test, date = 50, '20250607'
pattern = r'VGON_nqd\d+_L\d+_(\d{8}_\d{6})_\d{3}.mat'
files_with_datetime = filter(extract_datetime, os.listdir('./mats'))
for name in sorted(files_with_datetime, key=extract_datetime):
    match = re.search(pattern, name)
    if match and compare_datetime(date, match.group(1)):
        path = f'./mats/{name}'
        load = loadmat(path)
        n_train = load['n_train'].item()
        if 'overlaps' not in load:
            theta = load['theta'].item()
            phase = load['phase'].item()
            n_layers = load['n_layers'].item()
            n_qudits = load['n_qudits'].item()
            batch_size = load['batch_size'].item()
            generating(n_layers, n_qudits, n_test, batch_size, theta, phase, path)
