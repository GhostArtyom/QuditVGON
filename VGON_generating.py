import os
import re
import torch
import numpy as np
import torch.nn as nn
import pennylane as qml
from typing import List
from scipy.io import loadmat
import torch.nn.functional as F
import torch.distributions as dists
from utils import fidelity, updatemat
from torch.utils.data import DataLoader
from qudit_mapping import symmetric_decoding
from qutrit_synthesis import NUM_PR, two_qutrit_unitary_synthesis

np.set_printoptions(precision=8, linewidth=200)
torch.set_printoptions(precision=8, linewidth=200)


def testing(n_qudits: int, batch_size: int, n_test: int):
    n_layers = 2
    beta = -1 / 3
    n_qubits = 2 * n_qudits
    n_samples = batch_size * n_test
    n_params = n_layers * (n_qudits - 1) * NUM_PR

    z_dim = 50
    list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1)
    h_dim = np.power(2, list_z).astype(int)

    dev = qml.device('default.qubit', n_qubits)
    if torch.cuda.is_available() and n_qubits >= 14:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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
        params = params.reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
        qml.layer(qutrit_symmetric_ansatz, n_layers, params)
        return qml.state()

    @qml.qnode(dev, interface='torch', diff_method='best')
    def circuit_expval(n_layers: int, params: torch.Tensor, Ham):
        params = params.reshape(n_layers, n_qudits - 1, NUM_PR, batch_size)
        qml.layer(qutrit_symmetric_ansatz, n_layers, params)
        return qml.expval(Ham)

    class VAE_Model(nn.Module):

        def __init__(self, n_params: int, z_dim: int, h_dim: List[int]):
            super(VAE_Model, self).__init__()
            # encoder
            self.e1 = nn.ModuleList([nn.Linear(n_params, h_dim[0], bias=True)])
            self.e1 += [nn.Linear(h_dim[i - 1], h_dim[i], bias=True) for i in range(1, len(h_dim))]
            self.e2 = nn.Linear(h_dim[-1], z_dim, bias=True)
            self.e3 = nn.Linear(h_dim[-1], z_dim, bias=True)
            # decoder
            self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1], bias=True)])
            self.d4 += [nn.Linear(h_dim[-i + 1], h_dim[-i], bias=True) for i in range(2, len(h_dim) + 1)]
            self.d5 = nn.Linear(h_dim[0], n_params, bias=True)

        def encoder(self, x):
            h = F.relu(self.e1[0](x))
            for i in range(1, len(self.e1)):
                h = F.relu(self.e1[i](h))
            mean = self.e2(h)
            log_var = self.e3(h)
            return mean, log_var

        def reparameterize(self, mean, log_var):
            eps = torch.randn(log_var.shape).to(device)
            std = torch.exp(log_var).pow(0.5)
            z = mean + std * eps
            return z

        def decoder(self, z):
            params = F.relu(self.d4[0](z))
            for i in range(1, len(self.d4)):
                params = F.relu(self.d4[i](params))
            params = self.d5(params)
            return params

        def forward(self, x):
            mean, log_var = self.encoder(x)
            z = self.reparameterize(mean, log_var)
            params = self.decoder(z)
            return params, mean, log_var

    state_dict = torch.load(f'{path}.pt', weights_only=True)
    model = VAE_Model(n_params, z_dim, h_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    Ham = Hamiltonian(n_qudits, beta)
    data_dist = dists.Uniform(0, 1).sample([n_samples, n_params])
    test_data = DataLoader(data_dist, batch_size=batch_size, shuffle=True, drop_last=True)
    ED_states = loadmat('./mats/ED_degeneracy.mat')[f'nqd{n_qudits}'][0, 1].T

    overlaps = np.empty([0, ED_states.shape[0]])
    decoded_states = np.empty([0, 3**n_qudits])
    for i, batch in enumerate(test_data):
        params, _, _ = model(batch.to(device))
        energy = circuit_expval(n_layers, params, Ham).detach().numpy()
        states = circuit_state(n_layers, params).detach().numpy()
        for state in states:
            decoded_state = symmetric_decoding(state, n_qudits)
            decoded_states = np.vstack((decoded_states, decoded_state))
            overlap = [fidelity(decoded_state, ED_state) for ED_state in ED_states]
            overlaps = np.vstack((overlaps, overlap))
        print(i + 1, f'{energy.mean().item():.8f}', energy)
    updatemat(f'{path}.mat', {'overlaps': overlaps})


batch_size, n_test = 10, 100
pattern = r'(VGON_nqd\d+_\d{8}_\d{6}).mat'
for name in os.listdir('./mats'):
    match = re.search(pattern, name)
    if match:
        path = f'./mats/{match.group(1)}'
        load = loadmat(f'{path}.mat')
        n_qudits = int(re.search(r'nqd(\d+)', path).group(1))
        if 'overlaps' not in load:
            print(path)
            testing(n_qudits, batch_size, n_test)
