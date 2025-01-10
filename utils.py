import numpy as np
from pysr import PySRRegressor
from copy import deepcopy
import sympy

BINOPS = ["+", "-", "max", "min"]

def remove_constant(all_states):
    ncells = len(all_states[0])
    to_remove = []
    for i in range(ncells):
        if np.all(all_states[:, i] == all_states[0, i]):
            to_remove.append(i)

    states_poses = list(range(ncells))
    states = deepcopy(all_states)
    for i in reversed(to_remove):
        states_poses.remove(i)
        states = np.delete(states, i, axis=1)
    return states, states_poses

def get_model(l1_loss=True, min_val=None, max_val=None, binops=BINOPS):
    if l1_loss:
        loss = "loss(prediction, target) = abs(prediction - target)"
    else:
        loss = "loss(prediction, target) = (prediction - target)^2"

    un_ops = []
    extra_sympy_mappings = {}
    if min_val is not None:
        f = "max_" + str(min_val)
        un_ops.append(f + "(x) = max(x, " + str(min_val) + ")")
        extra_sympy_mappings[f] = lambda x: sympy.Max(min_val, x)
    if max_val is not None:
        f = "min_" + str(min_val)
        un_ops.append(f + "(x) = min(x, " + str(max_val) + ")")
        extra_sympy_mappings[f] = lambda x: sympy.Min(max_val, x)

    return PySRRegressor(
        niterations = 80,  # < Increase me for better results
        maxsize = 10,
        binary_operators = binops,
        unary_operators = un_ops,
        extra_sympy_mappings = extra_sympy_mappings,
        elementwise_loss = loss
    )

def split_constant_variable_rams(rams, next_rams, to_track):
    nstates, _ = rams.shape
    is_constant_at_state = np.zeros(nstates)
    for i, (state, next_state) in enumerate(zip(rams, next_rams)):
        if state[to_track] == next_state[to_track]:
            is_constant_at_state[i] = 1
    is_constant_at_state = is_constant_at_state.astype(np.bool_)

    non_cst_rams = rams[~is_constant_at_state]
    non_cst_next_rams = next_rams[~is_constant_at_state]
    return is_constant_at_state, non_cst_rams, non_cst_next_rams

def extend_with_signed_rams(rams):
    nstates, ncells = rams.shape
    extended_rams = np.zeros((nstates, 2 * ncells), dtype=int)
    extended_rams[:, :ncells] = rams
    for i, ram in enumerate(rams):
        signed_ram = ram.astype(np.int8)
        extended_rams[i, ncells:] = signed_ram
    return extended_rams