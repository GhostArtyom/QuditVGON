import time
import torch
import GPUtil
import numpy as np
import pennylane as qml
from logging import info
from logger import Logger
from VAE_model import VAEModel
from Hamiltonian import BBH_model
from itertools import combinations
import torch.distributions as dists
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader
from exact_diagonalization import qutrit_BBH_model
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)


def training(n_layers: int, n_qudits: int, n_iter: int, batch_size: int, theta: float, checkpoint: str = None):
    weight_decay = 0
    learning_rate = 1e-3
    n_qubits = 2 * n_qudits
    n_samples = batch_size * n_iter
    n_params = n_layers * (n_qudits - 1) * NUM_PR
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

    log = f'./logs/VGON_nqd{n_qudits}_L{n_layers}_phase_202502.log'
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
    def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
        params = params.transpose(0, 1).reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
        qml.layer(qutrit_symmetric_ansatz, n_layers, params)
        return qml.expval(Ham)

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

    data_dist = dists.Uniform(-np.pi, np.pi).sample([n_samples, n_params])
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

        cos_sims = torch.empty((0), device=device)
        for ind in combinations(range(batch_size), 2):
            cos_sim = torch.cosine_similarity(params[ind[0], :], params[ind[1], :], dim=0)
            cos_sims = torch.cat((cos_sims, cos_sim.unsqueeze(0)), dim=0)
        cos_sim_max = cos_sims.max()
        cos_sim_mean = cos_sims.mean()

        energy_coeff, kl_coeff, cos_sim_mean_coeff = 1, 1, 1
        if cos_sim_max > 0.9:
            cos_sim_max_coeff = 4
        elif cos_sim_max > 0.8:
            cos_sim_max_coeff = 3
        elif cos_sim_max > 0.7:
            cos_sim_max_coeff = 2
        # elif cos_sim_max > 0.6:
        #     cos_sim_max_coeff = 2
        else:
            cos_sim_max_coeff = 1
        loss = energy_coeff * energy_mean + kl_coeff * kl_div + cos_sim_mean_coeff * cos_sim_mean + cos_sim_max_coeff * cos_sim_max
        loss.backward()
        optimizer.step()

        t = time.perf_counter() - start
        cos_sim_str = f'Cos_Sim: {cos_sim_max_coeff}*{cos_sim_max.item():.8f}, {cos_sim_mean_coeff}*{cos_sim_mean.item():.8f}, {cos_sims.min().item():.8f}'
        info(f'Loss: {loss:.8f}, Energy: {energy_mean:.8f}, {energy_gap:.4e}, KL: {kl_div:.4e}, {cos_sim_str}, {i+1}/{n_iter}, {t:.2f}')

        energy_tol, kl_tol, cos_sim_tol = 1e-2, 1e-5, 0.6
        if i + 4 >= n_iter or (energy_gap < energy_tol and kl_div < kl_tol and cos_sim_max < cos_sim_tol):
            time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
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
                'energy': energy_mean.item(),
                'n_train': f'{i+1}/{n_iter}',
                'weight_decay': weight_decay,
                'learning_rate': learning_rate,
                'cos_sim_max': cos_sim_max.item(),
                'cos_sim_mean': cos_sim_mean.item(),
                'energy_iter': energy_iter.squeeze(),
                'ground_state_energy': ground_state_energy
            }
            savemat(f'{path}.mat', mat_dict)
            torch.save(model.state_dict(), f'{path}.pt')
            info(f'Save: {path}.mat&pt, {i+1}/{n_iter}')
    torch.cuda.empty_cache()
    logger.remove_handler()


n_qudits = 4
n_iter = 2000
batch_size = 8
# coeffs = np.array([-0.74, 0.49]) * np.pi

checkpoint = None
if checkpoint:
    load = loadmat(f'./mats/{checkpoint}.mat')
    coeffs = [load['theta'].item()]
    n_layers = load['n_layers'].item()
    n_qudits = load['n_qudits'].item()
    batch_size = load['batch_size'].item()

# n_layers = int(input('Input number of layers: '))
# for theta in coeffs:
theta = np.arctan(1 / 3)
for n_layers in [2, 3, 4]:
    training(n_layers, n_qudits, n_iter, batch_size, theta, checkpoint)
