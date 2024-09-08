import torch
import numpy as np
from utils import is_power_of_two
from scipy.sparse import csr_matrix

DTYPE = np.float64
CDTYPE = np.complex128


def symmetric_index(dim: int, n_qudits: int) -> dict:
    '''The index of the qudit state or matrix element corresponding to the qubit symmetric state or matrix during mapping.
    Args:
        dim (int): the dimension of qudit state or matrix.
        n_qudits (int): the number fo qudit state or matrix.
    Returns:
        ind (dict): which keys are the index of the qudit state or matrix,
        values are the corresponding index of qubit symmetric state or matrix.
    '''
    if n_qudits == 1:
        ind = {}
        for i in range(2**(dim - 1)):
            num1 = bin(i).count('1')
            if num1 in ind:
                ind[num1].append(i)
            else:
                ind[num1] = [i]
    else:
        ind, ind_ = {}, {}
        for i in range(2**(dim - 1)):
            num1 = bin(i).count('1')
            i_ = bin(i)[2::].zfill(dim - 1)
            if num1 in ind_:
                ind_[num1].append(i_)
            else:
                ind_[num1] = [i_]
        for i in range(dim**n_qudits):
            multi = ['']
            base = np.base_repr(i, dim).zfill(n_qudits)
            for j in range(n_qudits):
                multi = [x + y for x in multi for y in ind_[int(base[j])]]
            ind[i] = [int(x, 2) for x in multi]
    return ind


def is_symmetric(mat: np.ndarray, n_qudits: int = 1) -> bool:
    '''Check whether the qubit state or matrix is symmetric.
    Args:
        qubit (np.ndarray): the qubit state or matrix that needs to be checked whether it is symmetric.
        n_qubits (int): the number of qubits in the qubit symmetric state or matrix. Default: 1.
    Returns:
        is_sym (bool): whether the qubit state or matrix is symmetric.
    '''
    if mat.ndim == 2 and (mat.shape[0] == 1 or mat.shape[1] == 1):
        mat = mat.flatten()
    if mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
        raise ValueError(f'Wrong matrix shape {mat.shape}')
    if mat.ndim != 1 and mat.ndim != 2:
        raise ValueError(f'Wrong matrix shape {mat.shape}')
    is_sym = True
    n = mat.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f'Wrong matrix size {n} is not a power of 2')
    nq = int(np.log2(n))
    dim = nq // n_qudits + 1
    if nq % n_qudits == 0 and nq != n_qudits:
        ind = symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape} or n_qudits {n_qudits}')
    if mat.ndim == 1:
        for i in range(dim**n_qudits):
            i_ = ind[i]
            if len(i_) != 1:
                a = mat[i_]
                is_sym = is_sym & np.allclose(a, a[0])
    elif mat.ndim == 2:
        for i in range(dim**n_qudits):
            i_ = ind[i]
            for j in range(dim**n_qudits):
                j_ = ind[j]
                if len(i_) != 1 or len(j_) != 1:
                    a = mat[np.ix_(i_, j_)]
                    is_sym = is_sym & np.allclose(a, a[0][0])
    return is_sym


def symmetric_decoding(qubit: np.ndarray, n_qudits: int = 1) -> np.ndarray:
    '''Qudit symmetric state decoding, decodes a qubit symmetric state or matrix into a qudit state or matrix.
    Args:
        qubit (np.ndarray): the qubit symmetric state or matrix that needs to be decoded,
        where the qubit state or matrix must preserve symmetry.
        n_qudits (int): the number of qudits in the qudit state or matrix. Default: 1.
    Returns:
        qudit (np.ndarray): the qudit state or matrix obtained after the qudit symmetric decoding.
    '''
    if qubit.ndim == 2 and (qubit.shape[0] == 1 or qubit.shape[1] == 1):
        qubit = qubit.flatten()
    if qubit.ndim == 2 and qubit.shape[0] != qubit.shape[1]:
        raise ValueError(f'Wrong qubit state shape {qubit.shape}')
    if qubit.ndim != 1 and qubit.ndim != 2:
        raise ValueError(f'Wrong qubit state shape {qubit.shape}')
    n = qubit.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f'Wrong qubit state size {n} is not a power of 2')
    nq = int(np.log2(n))
    dim = nq // n_qudits + 1
    if nq % n_qudits == 0 and nq != n_qudits:
        ind = symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong qubit state shape {qubit.shape} or n_qudits {n_qudits}')
    if qubit.ndim == 1:
        qudit = np.zeros(dim**n_qudits, dtype=CDTYPE)
        for i in range(dim**n_qudits):
            i_ = ind[i]
            qubit_i = qubit[i_]
            if np.allclose(qubit_i, qubit_i[0]):
                qudit[i] = qubit_i[0] * np.sqrt(len(i_))
            else:
                raise ValueError('Qubit state is not symmetric')
    elif qubit.ndim == 2:
        qudit = np.zeros([dim**n_qudits, dim**n_qudits], dtype=CDTYPE)
        for i in range(dim**n_qudits):
            i_ = ind[i]
            for j in range(dim**n_qudits):
                j_ = ind[j]
                qubit_ij = qubit[np.ix_(i_, j_)]
                if np.allclose(qubit_ij, qubit_ij[0][0]):
                    div = np.sqrt(len(i_)) * np.sqrt(len(j_))
                    qudit[i, j] = qubit_ij[0][0] * div
                else:
                    raise ValueError('Qubit state is not symmetric')
    return qudit


def symmetric_encoding(qudit: np.ndarray, n_qudits: int = 1, is_csr: bool = False) -> np.ndarray:
    '''Qudit symmetric state encoding, encodes a qudit state or matrix into a qubit symmetric state or matrix.
    Args:
        qudit (np.ndarray): the qudit state or matrix that needs to be encoded.
        n_qudits (int): the number of qudits in the qudit state or matrix. Default: 1.
    Returns:
        qubit (np.ndarray): the qubit symmetric state or matrix obtained after the qudit symmetric encoding.
    '''
    if qudit.ndim == 2 and (qudit.shape[0] == 1 or qudit.shape[1] == 1):
        qudit = qudit.flatten()
    if qudit.ndim == 2 and qudit.shape[0] != qudit.shape[1]:
        raise ValueError(f'Wrong qudit state shape {qudit.shape}')
    if qudit.ndim != 1 and qudit.ndim != 2:
        raise ValueError(f'Wrong qudit state shape {qudit.shape}')
    dim = round(qudit.shape[0]**(1 / n_qudits), 12)
    if dim % 1 == 0:
        dim = int(dim)
        n = 2**((dim - 1) * n_qudits)
        ind = symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong qudit state shape {qudit.shape} or n_qudits {n_qudits}')
    if qudit.ndim == 1:
        qubit = csr_matrix((n, 1), dtype=CDTYPE)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            i_ = (ind_i, np.zeros(num_i))
            qubit += csr_matrix((data, i_), shape=(n, 1))
    elif qudit.ndim == 2:
        qubit = csr_matrix((n, n), dtype=CDTYPE)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            for j in range(dim**n_qudits):
                ind_j = ind[j]
                num_j = len(ind_j)
                i_ = np.repeat(ind_i, num_j)
                j_ = np.tile(ind_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubit += csr_matrix((data, (i_, j_)), shape=(n, n))
    if not is_csr:
        return qubit.toarray()
    return qubit
