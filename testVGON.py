import time
import torch
import numpy as np
import torch.nn as nn
import pennylane as qml
import torch.nn.functional as F
from itertools import combinations
import torch.distributions as dists

# setting/paras
nq = 7  # energy = -15
n_layers = 5
lim = -(nq - 2) * 3 + 0.1
batch_size = 4
n_samples = batch_size * 500
kl_coeff = 1
energy_coeff = 1
fidelity_coeff = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_dim = 50
shapes = [(n_layers, 2 * nq)]
n_params = sum([np.prod(shape) for shape in shapes])
list_z = np.arange(np.floor(np.log2(n_params)), np.ceil(np.log2(z_dim)) - 1, -1, dtype=int)
vec_exp = np.vectorize(lambda x: 2**x)
h_dim = vec_exp(list_z)


# target Hamiltonian
def Hamiltonian(nq):
    coeffs, obs = [], []
    for i in range(nq - 2):
        coeffs.extend([1, 1, 1])
        obs.extend([qml.PauliX(i) @ qml.PauliX(i + 1), qml.PauliY(i) @ qml.PauliY(i + 1), qml.PauliZ(i) @ qml.PauliZ(i + 1)])
        coeffs.extend([1, 1, 1])
        obs.extend([qml.PauliX(i) @ qml.PauliX(i + 2), qml.PauliY(i) @ qml.PauliY(i + 2), qml.PauliZ(i) @ qml.PauliZ(i + 2)])
        coeffs.extend([1, 1, 1])
        obs.extend([qml.PauliX(i + 1) @ qml.PauliX(i + 2), qml.PauliY(i + 1) @ qml.PauliY(i + 2), qml.PauliZ(i + 1) @ qml.PauliZ(i + 2)])
    return qml.Hamiltonian(coeffs, obs)


# PQC
def layer_hardware(inputs):
    for i in range(nq):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i + nq], wires=i)
    if nq > 1:
        for i in range(nq):
            qml.CZ(wires=[i, (i + 1) % nq])


dev = qml.device('default.qubit', wires=nq, shots=None)


@qml.qnode(dev, interface='torch', diff_method='best')
def circuit(inputs):
    qml.layer(layer_hardware, n_layers, inputs)
    return qml.expval(Hamiltonian(nq))  # False → PBC, True → OBC


@qml.qnode(dev, interface='torch', diff_method='best')
def circuit_vec(inputs):
    qml.layer(layer_hardware, n_layers, inputs)
    return qml.state()


# VAE model
class Model(nn.Module):

    def __init__(self, para_dim, z_dim, h_dim):
        super(Model, self).__init__()
        # encoder
        self.e1 = nn.ModuleList([nn.Linear(para_dim, h_dim[0], bias=True)])
        self.e1 += [nn.Linear(h_dim[i - 1], h_dim[i], bias=True) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim, bias=True)  # get mean prediction
        self.e3 = nn.Linear(h_dim[-1], z_dim, bias=True)  # get mean prediction

        # decoder
        self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1], bias=True)])
        self.d4 += [nn.Linear(h_dim[-i + 1], h_dim[-i], bias=True) for i in range(2, len(h_dim) + 1)]
        self.d5 = nn.Linear(h_dim[0], para_dim, bias=True)

    def encoder(self, x):
        h = F.relu(self.e1[0](x))
        for i in range(1, len(self.e1)):
            h = F.relu(self.e1[i](h))
        # get mean
        mean = self.e2(h)
        # get log of variance
        log_var = self.e3(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = torch.randn(log_var.shape).to(device)
        std = torch.exp(log_var).pow(0.5)  # square root
        z = mean + std * eps
        return z

    def decoder(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)
        return circuit(out.transpose(0, 1).reshape(tuple(np.append(shapes, batch_size))))

    def decoder_vec(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)
        return out

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder_vec(z)
        return out, mean, log_var


data_dist = dists.Uniform(0, 1)
x = data_dist.sample([n_samples, n_params])
train_db = torch.utils.data.DataLoader(list(x), shuffle=True, batch_size=batch_size, drop_last=True)

# model
model = Model(n_params, z_dim, h_dim)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.perf_counter()
for i, batch_x in enumerate(train_db):
    # training phase
    model.train()
    batch_x = batch_x.to(device)
    optimizer.zero_grad(set_to_none=True)
    out, mean, log_var = model(batch_x)

    # loss function
    # 1. energy
    out_energy = circuit(out.transpose(0, 1).reshape(tuple(np.append(shapes, batch_size))))

    # 2. kl
    kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kl_div = kl_div.mean()

    # 3. fidelity among a batch
    state = circuit_vec(out.transpose(0, 1).reshape(tuple(np.append(shapes, batch_size))))

    fidelities = torch.empty((0))
    for state_index in combinations(range(np.shape(state)[0]), 2):
        fidelity = qml.math.fidelity_statevector(state[state_index[0]], state[state_index[1]], check_state=False)
        fidelities = torch.cat((fidelities, fidelity.unsqueeze(0)), dim=0)
    fidelity_average = fidelities.mean()

    energy = torch.mean(out_energy)

    coeffs_list = [20, 20, 10, 5, 5, 2, 2, 2] + [1] * 20
    fidelity_coeff = coeffs_list[int(i // 100)]

    loss = energy_coeff * energy + kl_coeff * kl_div + fidelity_coeff * fidelity_average
    loss.backward()
    optimizer.step()

    t = time.perf_counter() - start
    print(f'Loss: {loss.item():.8f}, Energy: {energy.item():.8f}, KL: {kl_div.item():.8f}, Fidelity: {fidelity.item():.8f}, {fidelity_coeff}, {i+1}, {t:.2f}')
