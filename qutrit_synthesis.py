import torch
import numpy as np
import pennylane as qml
from typing import List

NUM_PR = 102
DTYPE = np.float64
CDTYPE = np.complex128


def circuit_ind(ind: List[int], obj: List[int]):
    '''The qubit circuit for different subspace indices of the qutrit two-level unitary gate.
    Args:
        ind (List[int]): the subspace index of the qutrit two-level unitary gate.
        obj (List[int]): object qubits of the circuit.
    '''
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


def circuit_zyz(pr: torch.Tensor | np.ndarray, obj: List[int]):
    '''The qubit circuit for ZYZ decomposition of the qutrit two-level unitary gate.
    Args:
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        obj (List[int]): object qubits of the circuit.
    '''
    qml.CRZ(pr[0], [obj[0], obj[1]])
    qml.CRY(pr[1], [obj[0], obj[1]])
    qml.CRZ(pr[2], [obj[0], obj[1]])


def circuit_ctrl_state(state: int, ctrl: List[int]):
    '''The qubit circuit for different control states of the qutrit controlled gate.
    Args:
        state (int): the control state of the qutrit controlled gate.
        ctrl (List[int]): control qubits of the circuit.
    '''
    if state == 0:
        qml.X(ctrl[1])
        qml.X(ctrl[2])
    elif state == 1:
        qml.CNOT([ctrl[2], ctrl[1]])
        qml.RY(np.pi / 2, ctrl[2])
    elif state == 2:
        pass


def circuit_ctrl_ind(ind: List[int], obj: List[int], ctrl: List[int]):
    '''The qubit circuit for different subspace indices of the qutrit controlled gate.
    Args:
        ind (List[int]): the subspace index of the qutrit controlled gate.
        obj (List[int]): object qubits of the circuit.
        ctrl (List[int]): control qubits of the circuit.
    '''
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


def circuit_ctrl_gate(name: str, pr: float, obj: int, ctrl: List[int]):
    '''The qubit circuit for different types of the qutrit controlled gate.
    Args:
        name (str): the name of the qutrit controlled gate.
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        obj (int): object qubit of the circuit.
        ctrl (List[int]): control qubits of the circuit.
    '''
    gate_list = ['RX', 'RY', 'RZ', 'PS']
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
        raise ValueError(f'The gate name {name} is not in {gate_list}')


def two_level_unitary_synthesis(pr: torch.Tensor | np.ndarray, ind: List[int], obj: List[int]):
    '''Synthesize a qutrit two-level unitary gate with a qubit circuit.
    Args:
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        ind (List[int]): the subspace index of the qutrit two-level unitary gate.
        obj (List[int]): object qubits of the circuit.
    '''
    if len(pr) != 3:
        raise ValueError(f'The number of params {len(pr)} should be 3')
    if len(ind) != 2:
        raise ValueError(f'The qutrit unitary index length {len(ind)} should be 2')
    if len(set(ind)) != len(ind):
        raise ValueError(f'The qutrit unitary index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) > 2:
        raise ValueError(f'The qutrit unitary index {ind} should in 0 to 2')
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    circuit_ind(ind, obj)
    circuit_zyz(pr, obj)
    qml.adjoint(circuit_ind, lazy=False)(ind, obj)


def single_qutrit_unitary_synthesis(pr: torch.Tensor | np.ndarray, obj: List[int]):
    '''Synthesize a single-qutrit unitary gate with a qubit circuit.
    Args:
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        obj (List[int]): object qubits of the circuit.
    '''
    if len(pr) != 9:
        raise ValueError(f'The number of params {len(pr)} should be 9')
    if len(obj) != 2:
        raise ValueError(f'The number of object qubits {len(obj)} should be 2')
    index = [[0, 1], [0, 2], [1, 2]]
    pr = pr.reshape(3, 3)
    for i, ind in enumerate(index):
        two_level_unitary_synthesis(pr[i], ind, obj)


def controlled_rotation_synthesis(pr: torch.Tensor | np.ndarray, state: int, obj: int, ctrl: List[int], ind: List[int], name: str = 'RY'):
    '''Synthesize a qutrit controlled rotation gate with a qubit circuit.
    Args:
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        state (int): the control state of the qutrit controlled rotation gate.
        obj (int): object qubit of the circuit.
        ctrl (List[int]): control qubits of the circuit.
        ind (List[int]): the subspace index of the qutrit controlled rotation gate.
        name (str): the name of controlled gate. Default: 'RY'.
    '''
    ctrl_state = [0, 1, 2]
    if state not in ctrl_state:
        raise ValueError(f'The control state is not in {ctrl_state}')
    if len(pr) != 1:
        raise ValueError(f'The number of params {len(pr)} should be 1')
    if len(ind) != 2:
        raise ValueError(f'The qutrit unitary index length {len(ind)} should be 2')
    if len(set(ind)) != len(ind):
        raise ValueError(f'The qutrit unitary index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) > 2:
        raise ValueError(f'The qutrit unitary index {ind} should in 0 to 2')
    circuit_ctrl_state(state, ctrl)
    circuit_ctrl_ind(ind, obj, ctrl)
    circuit_ctrl_gate(name, pr[0], obj, ctrl)
    qml.adjoint(circuit_ctrl_ind, lazy=False)(ind, obj, ctrl)
    qml.adjoint(circuit_ctrl_state, lazy=False)(state, ctrl)


def controlled_diagonal_synthesis(pr: torch.Tensor | np.ndarray, state: int, obj: int, ctrl: List[int]):
    '''Synthesize a qutrit controlled diagonal gate with a qubit circuit.
    Args:
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        state (int): the control state of the qutrit controlled diagonal gate.
        obj (int): object qubit of the circuit.
        ctrl (List[int]): control qubits of the circuit.
    '''
    ctrl_state = [0, 1, 2]
    if state not in ctrl_state:
        raise ValueError(f'The control state is not in {ctrl_state}')
    if len(pr) != 3:
        raise ValueError(f'The number of params {len(pr)} should be 3')
    circuit_ctrl_state(state, ctrl)
    circuit_ctrl_ind([0, 1], obj, ctrl)
    circuit_ctrl_gate('RZ', pr[0], obj, ctrl)
    qml.adjoint(circuit_ctrl_ind, lazy=False)([0, 1], obj, ctrl)
    circuit_ctrl_ind([0, 2], obj, ctrl)
    circuit_ctrl_gate('RZ', pr[1], obj, ctrl)
    qml.adjoint(circuit_ctrl_ind, lazy=False)([0, 2], obj, ctrl)
    circuit_ctrl_gate('PS', pr[2], ctrl[1], ctrl[2])
    qml.adjoint(circuit_ctrl_state, lazy=False)(state, ctrl)


def two_qutrit_unitary_synthesis(pr: torch.Tensor | np.ndarray, obj: List[int]):
    '''Synthesize a two-qutrit unitary gate with a qubit circuit.
    Args:
        pr (torch.Tensor|np.ndarray): the paramters of the circuit.
        obj (List[int]): object qubits of the circuit.
    '''
    if len(pr) != NUM_PR:
        raise ValueError(f'The number of params {len(pr)} should be {NUM_PR}')
    if len(obj) != 4:
        raise ValueError(f'The number of object qubits {len(obj)} should be 4')
    single_qutrit_unitary_synthesis(pr[0:9], obj[2:])  # U1 pr:9
    controlled_diagonal_synthesis(pr[9:12], 1, obj[-1], obj[::-1][1:])  # CD1 pr:3
    single_qutrit_unitary_synthesis(pr[12:21], obj[2:])  # U2 pr:9
    controlled_diagonal_synthesis(pr[21:24], 2, obj[-1], obj[::-1][1:])  # CD2 pr:3
    single_qutrit_unitary_synthesis(pr[24:33], obj[2:])  # U3 pr:9
    controlled_rotation_synthesis(pr[33:34], 1, obj[0], obj[1:], [1, 2])  # RY1_1 pr:1
    controlled_rotation_synthesis(pr[34:35], 2, obj[0], obj[1:], [1, 2])  # RY1_2 pr:1
    single_qutrit_unitary_synthesis(pr[35:44], obj[2:])  # U4 pr:9
    controlled_diagonal_synthesis(pr[44:47], 2, obj[-1], obj[::-1][1:])  # CD3 pr:3
    single_qutrit_unitary_synthesis(pr[47:56], obj[2:])  # U5 pr:9
    controlled_rotation_synthesis(pr[56:57], 1, obj[0], obj[1:], [0, 1])  # RY2_1 pr:1
    controlled_rotation_synthesis(pr[57:58], 2, obj[0], obj[1:], [0, 1])  # RY2_2 pr:1
    single_qutrit_unitary_synthesis(pr[58:67], obj[2:])  # U6 pr:9
    controlled_diagonal_synthesis(pr[67:70], 0, obj[-1], obj[::-1][1:])  # CD4 pr:3
    single_qutrit_unitary_synthesis(pr[70:79], obj[2:])  # U7 pr:9
    controlled_rotation_synthesis(pr[79:80], 1, obj[0], obj[1:], [1, 2])  # RY3_1 pr:1
    controlled_rotation_synthesis(pr[80:81], 2, obj[0], obj[1:], [1, 2])  # RY3_2 pr:1
    single_qutrit_unitary_synthesis(pr[81:90], obj[2:])  # U8 pr:9
    controlled_diagonal_synthesis(pr[90:93], 2, obj[-1], obj[::-1][1:])  # CD5 pr:3
    single_qutrit_unitary_synthesis(pr[93:102], obj[2:])  # U9 pr:9
