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
from VAE_model import VAEModel
from Hamiltonian import AKLT_model
import torch.distributions as dists
from utils import fidelity, updatemat
from torch.utils.data import DataLoader
from qudit_mapping import symmetric_decoding
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)


def testing(batch_size: int, n_test: int, energy_upper: float):
    n_samples = batch_size * n_test
    n_params = n_layers * (n_qudits - 1) * NUM_PR
    list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1)
    h_dim = np.power(2, list_z).astype(int)

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

    state_dict = torch.load(f'{path}.pt', map_location=device, weights_only=True)
    model = VAEModel(n_params, z_dim, h_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_dist = dists.Uniform(0, 1).sample([n_samples, n_params])
    test_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)

    ED_states = loadmat('./mats/ED_degeneracy.mat')[f'nqd{n_qudits}'][0, 1].T
    ED_states[np.abs(ED_states) < 1e-15] = 0
    overlaps = np.empty([0, ED_states.shape[0]])
    Ham = AKLT_model(n_qudits, beta)

    count, count_str = 0, ''
    start = time.perf_counter()
    for i, batch in enumerate(test_data):
        for j in range(n_test):
            with torch.no_grad():
                params, _, _ = model(batch.to(device))
            energy = circuit_expval(n_layers, params, Ham)
            energy_str = f'Energy: {energy.max():.8f}, {energy.mean():.8f}, {energy.min():.8f}'
            energy_ind = torch.where(energy < energy_upper)[0]
            if len(energy_ind) >= 0.75 * batch_size:
                count += len(energy_ind)
                count_str = f'{count}/{(i+1)*batch_size}'
                break
            t = time.perf_counter() - start
            info(f'{energy_str}, {len(energy_ind)}<{0.75*batch_size:.0f}, {j+1}/{i+1}/{n_test}, {t:.2f}')

        states = circuit_state(n_layers, params)
        states = states[energy_ind].detach().cpu().numpy()
        for state in states:
            decoded_state = symmetric_decoding(state, n_qudits)
            overlap = [fidelity(decoded_state, ED_state) for ED_state in ED_states]
            overlaps = np.vstack((overlaps, overlap))

        t = time.perf_counter() - start
        info(f'{energy_str}, {count_str}, {i+1}/{n_test}, {t:.2f}')

    updatemat(f'{path}.mat', {'count': count_str, 'overlaps': overlaps})
    info(f'Save: {path}.mat with count and overlaps')


z_dim = 50
n_layers = 2
n_qudits = 7
beta = -1 / 3
n_qubits = 2 * n_qudits
ground_state_energy = -2 / 3 * (n_qudits - 1)

dev = qml.device('default.qubit', n_qubits)
gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
if torch.cuda.is_available() and gpu_memory < 0.5 and n_qubits >= 12:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

log = f'./logs/VGON_nqd{n_qudits}_generating_202412.log'
logger = Logger(log)
logger.add_handler()

info(f'PyTorch Device: {device}')
info(f'Number of qudits: {n_qudits}')
info(f'Number of qubits: {n_qubits}')
info(f'Ground State Energy: {ground_state_energy:.4f}')

pattern = f'(VGON_nqd{n_qudits}' + r'_\d{8}_\d{6}).mat'
for name in sorted(os.listdir('./mats'), reverse=True):
    match = re.search(pattern, name)
    if match:
        path = f'./mats/{match.group(1)}'
        load = loadmat(f'{path}.mat')
        if 'overlaps' not in load:
            energy = load['energy'].item()
            kl_div = load['kl_div'].item()
            batch_size = load['batch_size'].item()
            if energy > -3.9:
                energy_upper = -3
            elif energy > -3.95:
                energy_upper = -3.9
            elif energy > -3.99:
                energy_upper = -3.95
            else:
                energy_upper = -3.99
            n_test = 100 if batch_size == 16 else int(input('Input number of test: '))
            if 'fidelity_max' in load and 'fidelity_mean' in load:
                fidelity_max = load['fidelity_max'].item()
                fidelity_mean = load['fidelity_mean'].item()
                fidelity_str = f'Fidelity: {fidelity_max:.8f}, {fidelity_mean:.8f}'
                # if energy < -3.99 and fidelity_max < 0.98:
                logger.add_handler()
                n_train = load['n_train'].item()
                info(f'Load: {path}.mat, {n_train}')
                info(f'Energy: {energy:.8f}, Energy Upper: {energy_upper:.2f}, KL: {kl_div:.4e}, {fidelity_str}')
                testing(batch_size, n_test, energy_upper)
                logger.remove_handler()
