import sys
import logging
import numpy as np
from logging import info
from scipy.sparse.linalg import eigsh
from utils import tensor_product_sparse
from scipy.sparse import eye, csr_matrix
from qudit_mapping import CDTYPE, symmetric_encoding

np.set_printoptions(precision=8, linewidth=200)


def parity(num: int) -> str:
    if (num % 2) == 0:
        return 'even'
    return ' odd'


def qutrit_spin_operator(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((3**n_qudits, 3**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 3**i, 3**(n_qudits - i - 2)
        for j in s_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    if is_csr:
        return ss
    return ss.toarray()


def qutrit_spin_operator2(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((3**n_qudits, 3**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 3**i, 3**(n_qudits - i - 2)
        for j in s2_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    if is_csr:
        return ss
    return ss.toarray()


def qutrit_Hamiltonian(n_qudits: int, beta: float, is_csr: bool = False):
    s1 = qutrit_spin_operator(n_qudits, is_csr=True)
    s2 = qutrit_spin_operator2(n_qudits, is_csr=True)
    Ham = s1 - beta * s2
    if is_csr:
        return Ham
    return Ham.toarray()


def qubit_spin_operator(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((4**n_qudits, 4**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 4**i, 4**(n_qudits - i - 2)
        for j in sym_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    if is_csr:
        return ss
    return ss.toarray()


def qubit_spin_operator2(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((4**n_qudits, 4**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 4**i, 4**(n_qudits - i - 2)
        for j in sym2_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    if is_csr:
        return ss
    return ss.toarray()


def qubit_Hamiltonian(n_qudits: int, beta: float, is_csr: bool = False):
    s1 = qubit_spin_operator(n_qudits, is_csr=True)
    s2 = qubit_spin_operator2(n_qudits, is_csr=True)
    if is_csr:
        return s1 - beta * s2
    return (s1 - beta * s2).toarray()


log = './logs/ED_degeneracy.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

sx = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
sy = csr_matrix([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
sz = csr_matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
s_list = [sx, sy, sz]
s2_list = [i @ j for i in s_list for j in s_list]
sym_list = [symmetric_encoding(i, is_csr=True) for i in s_list]
sym2_list = [i @ j for i in sym_list for j in sym_list]

for beta in [-0.3, -0.4]:
    info(f'Coefficient beta: {beta:.4f}')
    for n_qudits in range(2, 11):
        n_qubits = 2 * n_qudits
        h3 = qutrit_Hamiltonian(n_qudits, beta, is_csr=True)
        h2 = qubit_Hamiltonian(n_qudits, beta, is_csr=True)
        v3 = np.sort(eigsh(h3, k=5, which='SA', return_eigenvectors=False))
        v2 = np.sort(eigsh(h2, k=5, which='SA', return_eigenvectors=False))
        info(f'nqd: {n_qudits:2d} {parity(n_qudits)} {v3}')
        info(f'nqb: {n_qubits:2d} {parity(n_qudits)} {v2}')
