import numpy as np
from pysr import PySRRegressor
from copy import deepcopy

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

def get_model():
    model = PySRRegressor(
        niterations = 100,  # < Increase me for better results
        maxsize = 15,
        binary_operators = ["+", "-", "max", "min"],
        elementwise_loss = "loss(prediction, target) = (prediction - target)^2",
    )
    return model