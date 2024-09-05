import numpy as np
import pennylane as qml
from typing import List


def circ_ind(ind: List[int], obj: List[int]):
    if ind == [0, 1]:
        qml.CNOT([obj[0], obj[1]])
        qml.CRY(-np.pi / 2, [obj[1], obj[0]])
        qml.X(obj[0])
    elif ind == [0, 2]:
        qml.CNOT([obj[1], obj[0]])
        qml.X(obj[0])
    elif ind == [1, 2]:
        qml.CNOT([obj[0], obj[1]])
        qml.CRY(np.pi / 2, [obj[1], obj[0]])
        qml.X(obj[1])


def circ_zyz(pr, obj: List[int]):
    qml.CRZ(pr[0], [obj[0], obj[1]])
    qml.CRY(pr[1], [obj[0], obj[1]])
    qml.CRZ(pr[2], [obj[0], obj[1]])


def circ_ctrl_state(state: int, ctrl: List[int]):
    if state == 0:
        qml.X(ctrl[1])
        qml.X(ctrl[2])
    elif state == 1:
        qml.CNOT([ctrl[2], ctrl[1]])
        qml.RY(np.pi / 2, ctrl[2])
    elif state == 2:
        pass


def circ_ctrl_ind(ind: List[int], obj: List[int], ctrl: List[int]):
    if ind == [0, 1]:
        qml.ctrl(qml.X(obj), control=ctrl)
        qml.ctrl(qml.RY(-np.pi / 2, ctrl[0]), [obj] + ctrl[1:])
        qml.ctrl(qml.X(ctrl[0]), control=ctrl[1:])
    elif ind == [0, 2]:
        qml.ctrl(qml.X(ctrl[0]), ctrl[1:] + [obj])
        qml.ctrl(qml.X(ctrl[0]), ctrl[1:])
    elif ind == [1, 2]:
        qml.ctrl(qml.X(obj), ctrl)
        qml.ctrl(qml.RY(np.pi / 2, ctrl[0]), [obj] + ctrl[1:])
        qml.ctrl(qml.X(obj), ctrl[1:])


def circ_ctrl_gate(name: str, pr: float, obj: int, ctrl: List[int]):
    name = name.upper()
    if 'RX' in name:
        qml.ctrl(qml.RX(pr, obj), ctrl)
    elif 'RY' in name:
        qml.ctrl(qml.RY(pr, obj), ctrl)
    elif 'RZ' in name:
        qml.ctrl(qml.RZ(pr, obj), ctrl)
    elif 'PS' in name:
        qml.ctrl(qml.PhaseShift(pr, obj), ctrl)
    else:
        raise ValueError(f'Wrong input name of rotation gate {name}')


def two_level_unitary_synthesis(dim: int, pr, ind: List[int], obj: List[int]):
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    if len(pr) != 3:
        raise ValueError(f'The number of params {len(pr)} should be 3')
    if len(ind) != 2:
        raise ValueError(f'The qutrit unitary index length {len(ind)} should be 2')
    if len(set(ind)) != len(ind):
        raise ValueError(f'The qutrit unitary index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) >= dim:
        raise ValueError(f'The qutrit unitary index {ind} should in 0 to {dim-1}')
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    circ_ind(ind, obj)
    circ_zyz(pr, obj)
    qml.adjoint(circ_ind, lazy=False)(ind, obj)


def single_qutrit_unitary_synthesis(dim: int, pr, obj: List[int]):
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    if len(pr) != 9:
        raise ValueError(f'The number of params {len(pr)} should be 9')
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    index = [[0, 1], [0, 2], [1, 2]]
    pr = pr.reshape(3, 3)
    for i, ind in enumerate(index):
        two_level_unitary_synthesis(dim, pr[i], ind, obj)


def controlled_rotation_synthesis(dim: int, pr, state: int, obj: int, ctrl: List[int], ind: List[int], name: str = 'RY'):
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    ctrl_state = list(range(dim))
    if state not in ctrl_state:
        raise ValueError(f'The control state is not in {ctrl_state}')
    if len(pr) != 1:
        raise ValueError(f'The number of params {len(pr)} should be 1')
    if len(ind) != 2:
        raise ValueError(f'The qutrit unitary index length {len(ind)} should be 2')
    if len(set(ind)) != len(ind):
        raise ValueError(f'The qutrit unitary index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) >= dim:
        raise ValueError(f'The qutrit unitary index {ind} should in 0 to {dim-1}')
    circ_ctrl_state(state, ctrl)
    circ_ctrl_ind(ind, obj, ctrl)
    circ_ctrl_gate(name, pr[0], obj, ctrl)
    qml.adjoint(circ_ctrl_ind, lazy=False)(ind, obj, ctrl)
    qml.adjoint(circ_ctrl_state, lazy=False)(state, ctrl)


def controlled_diagonal_synthesis(dim: int, pr, state: int, obj: int, ctrl: List[int]):
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    ctrl_state = list(range(dim))
    if state not in ctrl_state:
        raise ValueError(f'The control state is not in {ctrl_state}')
    if len(pr) != 3:
        raise ValueError(f'The number of params {len(pr)} should be 3')
    circ_ctrl_state(state, ctrl)
    circ_ctrl_ind([0, 1], obj, ctrl)
    circ_ctrl_gate('RZ', pr[0], obj, ctrl)
    qml.adjoint(circ_ctrl_ind, lazy=False)([0, 1], obj, ctrl)
    circ_ctrl_ind([0, 2], obj, ctrl)
    circ_ctrl_gate('RZ', pr[1], obj, ctrl)
    qml.adjoint(circ_ctrl_ind, lazy=False)([0, 2], obj, ctrl)
    circ_ctrl_gate('PS', pr[2], ctrl[1], ctrl[2])
    qml.adjoint(circ_ctrl_state, lazy=False)(state, ctrl)


def two_qutrit_unitary_synthesis(dim: int, pr, obj: List[int]):
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    if len(pr) != 102:
        raise ValueError(f'The number of params {len(pr)} should be 102')
    if len(obj) != 4:
        raise ValueError(f'The number of object qubits {len(obj)} should be 4')
    single_qutrit_unitary_synthesis(dim, pr[0:9], obj[2:])  # U1 pr:9
    controlled_diagonal_synthesis(dim, pr[9:12], 1, obj[-1], obj[::-1][1:])  # CD1 pr:3
    single_qutrit_unitary_synthesis(dim, pr[12:21], obj[2:])  # U2 pr:9
    controlled_diagonal_synthesis(dim, pr[21:24], 2, obj[-1], obj[::-1][1:])  # CD2 pr:3
    single_qutrit_unitary_synthesis(dim, pr[24:33], obj[2:])  # U3 pr:9
    controlled_rotation_synthesis(dim, pr[33:34], 1, obj[0], obj[1:], [1, 2])  # RY1_1 pr:1
    controlled_rotation_synthesis(dim, pr[34:35], 2, obj[0], obj[1:], [1, 2])  # RY1_2 pr:1
    single_qutrit_unitary_synthesis(dim, pr[35:44], obj[2:])  # U4 pr:9
    controlled_diagonal_synthesis(dim, pr[44:47], 2, obj[-1], obj[::-1][1:])  # CD3 pr:3
    single_qutrit_unitary_synthesis(dim, pr[47:56], obj[2:])  # U5 pr:9
    controlled_rotation_synthesis(dim, pr[56:57], 1, obj[0], obj[1:], [0, 1])  # RY2_1 pr:1
    controlled_rotation_synthesis(dim, pr[57:58], 2, obj[0], obj[1:], [0, 1])  # RY2_2 pr:1
    single_qutrit_unitary_synthesis(dim, pr[58:67], obj[2:])  # U6 pr:9
    controlled_diagonal_synthesis(dim, pr[67:70], 0, obj[-1], obj[::-1][1:])  # CD4 pr:3
    single_qutrit_unitary_synthesis(dim, pr[70:79], obj[2:])  # U7 pr:9
    controlled_rotation_synthesis(dim, pr[79:80], 1, obj[0], obj[1:], [1, 2])  # RY3_1 pr:1
    controlled_rotation_synthesis(dim, pr[80:81], 2, obj[0], obj[1:], [1, 2])  # RY3_2 pr:1
    single_qutrit_unitary_synthesis(dim, pr[81:90], obj[2:])  # U8 pr:9
    controlled_diagonal_synthesis(dim, pr[90:93], 2, obj[-1], obj[::-1][1:])  # CD5 pr:3
    single_qutrit_unitary_synthesis(dim, pr[93:102], obj[2:])  # U9 pr:9