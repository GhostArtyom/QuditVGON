import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from typing import Literal
from datetime import datetime
from VAE_model import VAEModel
from Hamiltonian import BBH_model
from itertools import combinations
import torch.distributions as dists
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader
from exact_diagonalization import qutrit_BBH_model
from qutrit_synthesis import single_qutrit_unitary_synthesis, controlled_diagonal_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)


def training(n_layers: int, n_qudits: int, n_iter: int, batch_size: int, theta: float, checkpoint: str = None):
    save_num = 4
    weight_decay = 0
    learning_rate = 1e-3
    n_qubits = 2 * n_qudits
    n_samples = batch_size * n_iter
    n_params_per_layer = 12 * n_qudits - 3
    n_params = n_layers * n_params_per_layer
    phase = 'arctan(1/3)' if theta == np.arctan(1 / 3) else f'{theta/np.pi:.2f}π'

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
    log = f'./logs/VGON_nqd{n_qudits}_L{n_layers}_phase_{year_month}.log'
    logger = Logger(log)
    logger.add_handler()

    info(f'PyTorch Device: {device}')
    info(f'Number of layers: {n_layers}')
    info(f'Number of qudits: {n_qudits}')
    info(f'Number of qubits: {n_qubits}')
    info(f'Number of params: {n_params}')
    info(f'Weight Decay: {weight_decay:.0e}')
    info(f'Learning Rate: {learning_rate:.0e}')
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
    def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
        params = params.transpose(0, 1).reshape(n_layers, n_params_per_layer, batch_size)
        qml.layer(qutrit_symmetric_ansatz, n_layers, params)
        return qml.expval(Ham)

    def metric(p: torch.Tensor, q: torch.Tensor, metric_type: Literal['Euclidean', 'Cosine Similarity', 'JS Divergence', 'Hellinger Distance']):
        if p.shape != q.shape:
            raise ValueError(f'Shapes of p {p.shape} and q {q.shape} must match')
        if metric_type == 'Euclidean':
            return torch.norm(p - q, 2)
        elif metric_type == 'Cosine Similarity':
            return torch.cosine_similarity(p, q, dim=0)
        elif metric_type == 'JS Divergence':
            p = torch.softmax(p, dim=0)
            q = torch.softmax(q, dim=0)
            m = (p + q) / 2
            return ((p * p.log2() + q * q.log2()) / 2 - m * m.log2()).sum()
        elif metric_type == 'Hellinger Distance':
            p = torch.softmax(p, dim=0)
            q = torch.softmax(q, dim=0)
            return ((p.sqrt() - q.sqrt()).pow(2).sum() / 2).sqrt()
        else:
            raise ValueError(f'Invalid metric_type: {metric_type}')

    qubit_Ham = BBH_model(n_qudits, theta)
    qutrit_Ham = qutrit_BBH_model(n_qudits, theta)

    ground_state_energy = eigsh(qutrit_Ham, k=4, which='SA', return_eigenvectors=False)
    ground_state_energy = ground_state_energy.min()
    info(f'Ground State Energy: {ground_state_energy:.8f}')

    model = VAEModel(n_params, h_dim, z_dim).to(device)
    if checkpoint:
        state_dict = torch.load(f'./mats/{checkpoint}.pt', map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        info(f'Load: state_dict of {checkpoint}.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_dist = dists.Uniform(0, 1).sample([n_samples, n_params])
    train_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)

    start = time.perf_counter()
    energy_iter = np.empty((0, batch_size))
    for i, batch in enumerate(train_data):
        optimizer.zero_grad(set_to_none=True)
        params, mean, log_var = model(batch.to(device))

        energy = circuit_expval(n_layers, params, qubit_Ham)
        energy_mean = energy.mean()
        energy = energy.detach().cpu().numpy()
        energy_iter = np.vstack((energy_iter, energy))
        energy_gap = energy_mean - ground_state_energy

        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kl_div = kl_div.mean()

        similarity_metrics = torch.empty((0), device=device)
        for ind in combinations(range(batch_size), 2):
            similarity_metric = metric(params[ind[0], :], params[ind[1], :], 'Cosine Similarity')
            similarity_metrics = torch.cat((similarity_metrics, similarity_metric.unsqueeze(0)), dim=0)
        similarity_var = similarity_metrics.var()
        similarity_max = similarity_metrics.max()
        similarity_mean = similarity_metrics.mean()

        energy_coeff, kl_coeff = 1, 1
        if similarity_max > 0.9:
            similarity_max_coeff = 1
        elif similarity_max > 0.8:
            similarity_max_coeff = 0.8
        elif similarity_max > 0.7:
            similarity_max_coeff = 0.6
        elif similarity_max > 0.6:
            similarity_max_coeff = 0.4
        else:
            similarity_max_coeff = 0.2

        if similarity_mean > 0.9:
            similarity_mean_coeff = 2
        elif similarity_mean > 0.8:
            similarity_mean_coeff = 1.5
        elif similarity_mean > 0.7:
            similarity_mean_coeff = 1
        elif similarity_mean > 0.6:
            similarity_mean_coeff = 0.9
        elif similarity_mean > 0.5:
            similarity_mean_coeff = 0.8
        elif similarity_mean > 0.4:
            similarity_mean_coeff = 0.7
        elif similarity_mean > 0.3:
            similarity_mean_coeff = 0.6
        elif similarity_mean > 0.2:
            similarity_mean_coeff = 0.5
        else:
            similarity_mean_coeff = 0.4

        loss = energy_coeff * energy_mean + kl_coeff * kl_div + similarity_max_coeff * similarity_max + similarity_mean_coeff * similarity_mean
        loss.backward()
        optimizer.step()

        t = time.perf_counter() - start
        similarity_str = f'Similarity: {similarity_max_coeff:.1f}*{similarity_max:.8f}, {similarity_mean_coeff:.1f}*{similarity_mean:.8f}, {similarity_var:.4e}'
        info(f'Loss: {loss:.8f}, Energy: {energy_mean:.8f}, {energy_gap:.4e}, KL: {kl_div:.4e}, {similarity_str}, {i+1}/{n_iter}, {t:.2f}')

        energy_tol, similarity_max_tol, similarity_mean_tol = 0.1, 0.6, 0.1
        if (i + save_num) >= n_iter or (energy_gap < energy_tol and similarity_max < similarity_max_tol and similarity_mean < similarity_mean_tol):
            save_num -= 1
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            path = f'./mats/VGON_nqd{n_qudits}_L{n_layers}_{time_str}'
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
                'energy_iter': energy_iter,
                'energy': energy_mean.item(),
                'n_train': f'{i+1}/{n_iter}',
                'weight_decay': weight_decay,
                'learning_rate': learning_rate,
                'similarity_var': similarity_var.item(),
                'similarity_max': similarity_max.item(),
                'similarity_mean': similarity_mean.item(),
                'ground_state_energy': ground_state_energy,
                'similarity_metrics': similarity_metrics.detach().cpu()
            }
            savemat(f'{path}.mat', mat_dict)
            torch.save(model.state_dict(), f'{path}.pt')
            info(f'Save: {path}.mat&pt, {i+1}/{n_iter}')
    torch.cuda.empty_cache()
    logger.remove_handler()


n_qudits = 4
n_iter = 2000
batch_size = 8
coeffs = [np.arctan(1 / 3)]

checkpoint = None
if checkpoint:
    load = loadmat(f'./mats/{checkpoint}.mat')
    coeffs = [load['theta'].item()]
    n_layers = load['n_layers'].item()
    n_qudits = load['n_qudits'].item()
    batch_size = load['batch_size'].item()

for theta in coeffs:
    for n_layers in [6, 7, 8]:
        training(n_layers, n_qudits, n_iter, batch_size, theta, checkpoint)
