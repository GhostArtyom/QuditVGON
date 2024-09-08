import os
import numpy as np
from math import log
from typing import List
from functools import reduce
from numpy.linalg import norm
from fractions import Fraction
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from scipy.io import loadmat, savemat

DTYPE = np.float64
CDTYPE = np.complex128


def updatemat(name: str, save: dict):
    '''Update the mat file.'''
    if os.path.exists(name):
        load = loadmat(name)
        load.update(save)
        savemat(name, load)
    else:
        savemat(name, save)


def dict_file(path: str) -> dict:
    '''Return a dict of the file path.'''
    dict_file = {}
    for root, _, files in os.walk(path):
        i = 1
        for name in sorted(files):
            subfolder = os.path.split(root)[-1]
            dict_file[f'{subfolder}_{i}'] = name
            i += 1
    return dict_file


def is_power_of_two(num: int) -> bool:
    '''Check if the number is a power of 2.'''
    return (num & (num - 1) == 0) and num != 0


def is_unitary(mat: np.ndarray) -> bool:
    '''Check if the matrix is ​​unitary.'''
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        dim = mat.shape[0]
        return np.allclose(np.eye(dim), mat @ mat.conj().T)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def is_hermitian(mat: np.ndarray) -> bool:
    '''Check if the matrix is hermitian.'''
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        return np.allclose(mat, mat.conj().T)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def approx_matrix(mat: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    '''Return an approximation of the matrix.'''
    if np.iscomplexobj(mat):
        mat_real = np.real(mat)
        mat_imag = np.imag(mat)
        mat_real[np.abs(mat_real) < tol] = 0
        mat_imag[np.abs(mat_imag) < tol] = 0
        mat = mat_real + 1j * mat_imag
        return mat_real if np.all(mat_imag == 0) else mat
    mat[np.abs(mat) < tol] = 0
    return mat


def random_qudit(dim: int, ndim: int = 1) -> np.ndarray:
    '''Generate a random one-qudit state or matrix.'''
    if ndim == 1:
        qudit = np.random.rand(dim) + 1j * np.random.rand(dim)
    elif ndim == 2:
        qudit = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    else:
        raise ValueError(f'Wrong qudit ndim {ndim}')
    qudit /= norm(qudit)
    return qudit


def random_qudits(dim: int, n_qudits: int, ndim: int = 1) -> np.ndarray:
    '''Generate a random n-qudit state or matrix by tensor product.'''
    qudit_list = [random_qudit(dim, ndim) for _ in range(n_qudits)]
    qudits = reduce(np.kron, qudit_list)
    qudits /= norm(qudits)
    return qudits


def str_special(param: str | int | float) -> str:
    '''Return a special string form of the parameter.'''
    special = {'': 1, 'π': np.pi, '√2': np.sqrt(2), '√3': np.sqrt(3), '√5': np.sqrt(5)}
    if isinstance(param, (int, str)):
        return str(param)
    elif param % 1 == 0:
        return str(int(param))
    coeff = -1 if param < 0 else 1
    param *= -1 if param < 0 else 1
    for k, v in special.items():
        frac = Fraction(param / v).limit_denominator(100)
        multi = round(param / v, 4)
        divisor = round(v / param, 4)
        if np.isclose(multi % 1, 0):
            coeff *= int(multi)
            param = k if coeff == 1 else f'-{k}' if coeff == -1 else f'{coeff}{k}'
            break
        elif np.isclose(divisor % 1, 0):
            coeff *= int(divisor)
            k = 1 if v == 1 else k
            param = f'{k}/{coeff}' if coeff > 0 else f'-{k}/{-coeff}'
            break
        elif abs(param / v - frac) < 1e-6:
            x, y = frac.numerator, frac.denominator
            x = '' if x == 1 else x
            param = f'{x}{k}/{y}' if coeff > 0 else f'-{x}{k}/{y}'
            break
    if isinstance(param, str):
        return param
    return str(round(param * coeff, 4))


def str_ket(state: np.ndarray, dim: int = 2, tol: float = 1e-8) -> str:
    '''Return a ket format of the qudit state.'''
    if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
        state = state.flatten()
    if state.ndim != 1:
        raise ValueError(f'State requires a 1-D ndarray, but get {state.shape}')
    nq = round(log(len(state), dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong state shape {state.shape} is not a power of {dim}')
    nq = int(nq)
    string = []
    for ind, value in enumerate(state):
        base = np.base_repr(ind, dim).zfill(nq)
        real = np.real(value)
        imag = np.imag(value)
        real_str = str_special(real)
        imag_str = str_special(imag)
        if np.abs(value) < tol:
            continue
        if np.abs(real) < tol:
            string.append(f'{imag_str}j¦{base}⟩')
            continue
        if np.abs(imag) < tol:
            string.append(f'{real_str}¦{base}⟩')
            continue
        if imag_str.startswith('-'):
            string.append(f'{real_str}{imag_str}j¦{base}⟩')
        else:
            string.append(f'{real_str}+{imag_str}j¦{base}⟩')
    print('\n'.join(string))
    print(state)


def partial_trace(rho: np.ndarray, dim: int, ind: int) -> np.ndarray:
    '''Calculate the partial trace of the qudit state or matrix.'''
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if rho.ndim != 1 and rho.ndim != 2:
        raise ValueError(f'Wrong state shape {rho.shape}')
    n = rho.shape[0]
    m = n // dim
    if n == dim and rho.ndim == 1:
        return rho.conj() @ rho
    elif n == dim and rho.ndim == 2:
        return np.trace(rho)
    nq = round(log(m, dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong matrix size {n} is not a power of {dim}')
    nq = int(nq)
    if ind < 0 or ind > nq:
        raise ValueError(f'Wrong index {ind} is not in 0 to {nq}')
    pt = csr_matrix((m, m), dtype=CDTYPE)
    for k in range(dim):
        i_ = np.zeros(m, dtype=np.int64)
        for i in range(m):
            ii = np.base_repr(i, dim).zfill(nq)
            i_[i] = int(ii[:ind] + str(k) + ii[ind:], dim)
        psi = csr_matrix((np.ones(m), (np.arange(m), i_)), shape=(m, n))
        if rho.ndim == 1:
            temp = psi.dot(csr_matrix(rho).T)
            pt += temp.dot(temp.conj().T)
        elif rho.ndim == 2:
            pt += psi.dot(csr_matrix(rho)).dot(psi.conj().T)
    return pt.toarray()


def reduced_density_matrix(rho: np.ndarray, dim: int, position: List[int]) -> np.ndarray:
    '''Calculate the reduced density matrix of the qudit state or matrix.'''
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if rho.ndim != 1 and rho.ndim != 2:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if isinstance(position, (int, np.int64)):
        position = [position]
    n = rho.shape[0]
    nq = round(log(n, dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong matrix size {n} is not a power of {dim}')
    nq = int(nq)
    p = [x for x in range(nq) if x not in position]
    for ind in p[::-1]:
        rho = partial_trace(rho, dim, ind)
    return rho


def fidelity(rho: np.ndarray, sigma: np.ndarray, sqrt: bool = False) -> float:
    '''Calculate the fidelity of two qudit states.'''
    state = {'rho': rho, 'sigma': sigma}
    for key, mat in state.items():
        if mat.ndim == 2 and (mat.shape[0] == 1 or mat.shape[1] == 1):
            mat = mat.flatten()
            state[key] = mat
        if mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
            raise ValueError(f'Wrong {key} shape {mat.shape}')
        if mat.ndim != 1 and mat.ndim != 2:
            raise ValueError(f'Wrong {key} shape {mat.shape}')
    rho, sigma = state.values()
    if rho.shape[0] != sigma.shape[0]:
        raise ValueError(f'Mismatch state shape: rho {rho.shape}, sigma {sigma.shape}')
    if rho.ndim == 1 and sigma.ndim == 1:
        f = np.abs(rho.conj() @ sigma)
        return f if sqrt else f**2
    elif rho.ndim == 1 and sigma.ndim == 2:
        f = np.real(rho.conj() @ sigma @ rho)
        return np.sqrt(f) if sqrt else f
    elif rho.ndim == 2 and sigma.ndim == 1:
        f = np.real(sigma.conj() @ rho @ sigma)
        return np.sqrt(f) if sqrt else f
    elif rho.ndim == 2 and sigma.ndim == 2:
        f = np.real(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))
        return f if sqrt else f**2
    else:
        raise ValueError(f'Wrong state ndim: rho {rho.ndim}, sigma {sigma.ndim}')
