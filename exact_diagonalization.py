import numpy as np
from logging import info
from logger import Logger
from scipy.sparse.linalg import eigsh
from utils import tensor_product_sparse
from scipy.sparse import eye, csr_matrix
from qudit_mapping import CDTYPE, symmetric_encoding

np.set_printoptions(precision=8, linewidth=200)


def parity(num: int) -> str:
    if (num % 2) == 0:
        return 'even'
    return ' odd'


def qubit_spin_operator(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((4**n_qudits, 4**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 4**i, 4**(n_qudits - i - 2)
        for j in sym_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    return ss if is_csr else ss.toarray()


def qubit_spin_operator2(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((4**n_qudits, 4**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 4**i, 4**(n_qudits - i - 2)
        for j in sym2_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    return ss if is_csr else ss.toarray()


def qutrit_spin_operator(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((3**n_qudits, 3**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 3**i, 3**(n_qudits - i - 2)
        for j in s_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    return ss if is_csr else ss.toarray()


def qutrit_spin_operator2(n_qudits: int, is_csr: bool = False):
    ss = csr_matrix((3**n_qudits, 3**n_qudits), dtype=CDTYPE)
    for i in range(n_qudits - 1):
        d1, d2 = 3**i, 3**(n_qudits - i - 2)
        for j in s2_list:
            ss += tensor_product_sparse(eye(d1), j, j, eye(d2))
    return ss if is_csr else ss.toarray()


def qubit_AKLT_model(n_qudits: int, beta: float, is_csr: bool = False):
    s1 = qubit_spin_operator(n_qudits, is_csr)
    s2 = qubit_spin_operator2(n_qudits, is_csr)
    Ham = s1 - beta * s2
    return Ham if is_csr else Ham.toarray()


def qutrit_AKLT_model(n_qudits: int, beta: float, is_csr: bool = False):
    s1 = qutrit_spin_operator(n_qudits, is_csr)
    s2 = qutrit_spin_operator2(n_qudits, is_csr)
    Ham = s1 - beta * s2
    return Ham if is_csr else Ham.toarray()


def qubit_BBH_model(n_qudits: int, theta: float, is_csr: bool = False):
    s1 = qubit_spin_operator(n_qudits, is_csr)
    s2 = qubit_spin_operator2(n_qudits, is_csr)
    Ham = np.cos(theta) * s1 + np.sin(theta) * s2
    return Ham if is_csr else Ham.toarray()


def qutrit_BBH_model(n_qudits: int, theta: float, is_csr: bool = False):
    s1 = qutrit_spin_operator(n_qudits, is_csr)
    s2 = qutrit_spin_operator2(n_qudits, is_csr)
    Ham = np.cos(theta) * s1 + np.sin(theta) * s2
    return Ham if is_csr else Ham.toarray()


def eigensolver_AKLT_model(n_qudits: int, beta: float, k: int = 1):
    n_qubits = 2 * n_qudits
    qutrit_Ham = qutrit_AKLT_model(n_qudits, beta, is_csr=True)
    qubit_Ham = qubit_AKLT_model(n_qudits, beta, is_csr=True)
    qutrit_eigval = np.sort(eigsh(qutrit_Ham, k, which='SA', return_eigenvectors=False))
    qubit_eigval = np.sort(eigsh(qubit_Ham, k, which='SA', return_eigenvectors=False))
    info(f'nqd: {n_qudits:2d} {parity(n_qudits)} {qutrit_eigval}')
    info(f'nqb: {n_qubits:2d} {parity(n_qudits)} {qubit_eigval}')


def eigensolver_BBH_model(n_qudits: int, theta: float, k: int = 1):
    if theta == np.arctan(1 / 3):
        phase = 'arctan(1/3)'
    else:
        phase = f'{theta/np.pi:.2f}π'
    info(f'Coefficient phase: {phase}')
    n_qubits = 2 * n_qudits
    qutrit_Ham = qutrit_BBH_model(n_qudits, theta, is_csr=True)
    qubit_Ham = qubit_BBH_model(n_qudits, theta, is_csr=True)
    qutrit_eigval = np.sort(eigsh(qutrit_Ham, k, which='SA', return_eigenvectors=False))
    qubit_eigval = np.sort(eigsh(qubit_Ham, k, which='SA', return_eigenvectors=False))
    info(f'nqd: {n_qudits:2d} {qutrit_eigval}')
    info(f'nqb: {n_qubits:2d} {qubit_eigval}')


sx = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
sy = csr_matrix([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
sz = csr_matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
s_list = [sx, sy, sz]
s2_list = [i @ j for i in s_list for j in s_list]
sym_list = [symmetric_encoding(i, is_csr=True) for i in s_list]
sym2_list = [i @ j for i in sym_list for j in sym_list]

if __name__ == '__main__':
    log = './logs/exact_diagonalization.log'
    logger = Logger(log)
    coeffs = np.array([0.32, -0.71, -0.30, -0.16]) * np.pi
    coeffs = np.append(coeffs, np.arctan(1 / 3))
    n_qudits, k = 8, 8
    for theta in coeffs:
        logger.add_handler()
        eigensolver_BBH_model(n_qudits, theta, k)
        logger.remove_handler()
