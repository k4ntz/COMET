import numpy as np
from pysr import PySRRegressor
from copy import deepcopy
import sympy

def remove_constant(all_states):
    to_remove = []
    for i in range(128):
        if np.all(all_states[:, i] == all_states[0, i]):
            to_remove.append(i)

    states_poses = list(range(128))
    states = deepcopy(all_states)
    for i in reversed(to_remove):
        states_poses.remove(i)
        states = np.delete(states, i, axis=1)
    return states, states_poses

def get_model(l1_loss=True, min_val=None, max_val=None):
    if l1_loss:
        loss = "L1DistLoss()"
    else:
        loss = "L2DistLoss()"

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
        binary_operators = ["+", "-", "max", "min"],
        unary_operators = un_ops,
        extra_sympy_mappings = extra_sympy_mappings,
        elementwise_loss = loss
    )