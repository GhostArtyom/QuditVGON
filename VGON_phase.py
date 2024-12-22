import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from utils import fidelity
from scipy.linalg import orth
from VAE_model import VAEModel
from Hamiltonian import BBH_model
import torch.distributions as dists
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader
from qudit_mapping import symmetric_decoding
from exact_diagonalization import qutrit_BBH_model
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)


def training(n_layers: int, n_qudits: int, n_iter: int, batch_size: int, theta: float, checkpoint: str = None):
    weight_decay = 1e-2
    learning_rate = 1e-3
    n_qubits = 2 * n_qudits
    n_samples = batch_size * n_iter
    n_params = n_layers * (n_qudits - 1) * NUM_PR
    phase = 'arctan(1/3)' if theta == np.arctan(1 / 3) else f'{theta/np.pi:.2f}'

    z_dim = 50
    list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1)
    h_dim = np.power(2, list_z).astype(int)

    dev = qml.device('default.qubit', n_qubits)
    gpu_memory = gpus[0].memoryUtil if (gpus := GPUtil.getGPUs()) else 1
    if torch.cuda.is_available() and gpu_memory < 0.5 and n_qubits >= 12:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    log = f'./logs/VGON_nqd{n_qudits}_phase_L{n_layers}_202412.log'
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

    model = VAEModel(n_params, z_dim, h_dim).to(device)
    if checkpoint:
        state_dict = torch.load(f'./mats/{checkpoint}.pt', map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        info(f'Load: state_dict of {checkpoint}.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_dist = dists.Uniform(0, 1).sample([n_samples, n_params])
    train_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)

    start = time.perf_counter()
    energy_iter = np.empty((0, batch_size))
    fidelity_iter = np.empty((0, batch_size, degeneracy))
    for i, batch in enumerate(train_data):
        optimizer.zero_grad(set_to_none=True)
        params, mean, log_var = model(batch.to(device))

        energy = circuit_expval(n_layers, params, qubit_Ham)
        energy_mean = energy.mean()
        energy = energy.detach().cpu().numpy()
        energy_iter = np.vstack((energy_iter, energy))

        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kl_div = kl_div.mean()

        states = circuit_state(n_layers, params).detach().cpu().numpy()
        fidelities = np.empty((0, degeneracy))
        for state in states:
            decoded_state = symmetric_decoding(state, n_qudits)
            overlap = [fidelity(decoded_state, ground_state) for ground_state in ground_states]
            fidelities = np.vstack((fidelities, overlap))
        fidelity_iter = np.concatenate((fidelity_iter, fidelities[np.newaxis]))
        fidelity_mean = fidelities.mean(axis=0)

        energy_coeff, kl_coeff = 1, 1
        loss = energy_coeff * energy_mean + kl_coeff * kl_div
        loss.backward()
        optimizer.step()

        t = time.perf_counter() - start
        info(f'Loss: {loss:.8f}, Energy: {energy_mean:.8f}, KL: {kl_div:.4e}, Fidelity: {fidelity_mean.max():.8f}, {i+1}/{n_iter}, {t:.2f}')

        energy_gap = energy_mean - ground_state_energy
        if i + 1 >= n_iter:
            time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            path = f'./mats/VGON_nqd{n_qudits}_{time_str}'
            mat_dict = {
                'theta': theta,
                'phase': phase,
                'n_iter': n_iter,
                'loss': loss.item(),
                'n_layers': n_layers,
                'n_qudits': n_qudits,
                'n_qubits': n_qubits,
                'kl_div': kl_div.item(),
                'batch_size': batch_size,
                'fidelity': fidelity_mean,
                'energy': energy_mean.item(),
                'n_train': f'{i+1}/{n_iter}',
                'weight_decay': weight_decay,
                'learning_rate': learning_rate,
                'ground_states': ground_states,
                'energy_iter': energy_iter.squeeze(),
                'fidelity_iter': fidelity_iter.squeeze(),
                'ground_state_energy': ground_state_energy
            }
            savemat(f'{path}.mat', mat_dict)
            torch.save(model.state_dict(), f'{path}.pt')
            info(f'Energy Gap: {energy_gap:.4e}, KL: {kl_div:.4e}, {i+1}/{n_iter}, Save: {path}.mat&pt')
    torch.cuda.empty_cache()
    logger.remove_handler()


n_qudits = 7
n_iter = 1000
batch_size = 8

coeffs = np.array([-0.74, -0.26, -0.24, 0.24, 0.26, 0.49]) * np.pi
coeffs = [np.arctan(1 / 3)]

checkpoint = None
if checkpoint:
    load = loadmat(f'./mats/{checkpoint}.mat')
    coeffs = [load['theta'].item()]
    n_layers = load['n_layers'].item()
    n_qudits = load['n_qudits'].item()
    batch_size = load['batch_size'].item()

for theta in coeffs:
    for n_layers in [3, 2, 1]:
        training(n_layers, n_qudits, n_iter, batch_size, theta, checkpoint)
