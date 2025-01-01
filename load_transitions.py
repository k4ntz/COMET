from argparse import ArgumentParser
import os
import pickle as pkl
import random

random.seed(0)

parser = ArgumentParser()
parser.add_argument("-g", "--game", type=str, default="Pong")

args = parser.parse_args()

transitions = pkl.load(open(f'transitions/{args.game}.pkl', 'rb'))

ramst = [t[0][0] for t in transitions]
rgbst = [t[0][1] for t in transitions]
actions = [t[1] for t in transitions]
rewards = [t[2] for t in transitions]
terms = [t[3] for t in transitions]
truncs = [t[4] for t in transitions]


from ocatari.core import OCAtari

env = OCAtari(args.game)
actuples = []
for i, action_n in enumerate(actions):
    action = env.get_action_meanings()[action_n]
    act_tuple = [0, 0, 0] # x_axis, y_axis, button
    if "FIRE" in action:
        act_tuple[2] = 1
    if "LEFT" in action:
        act_tuple[0] = -1
    elif "RIGHT" in action:
        act_tuple[0] = 1
    if "DOWN" in action:
        act_tuple[1] = -1
    elif "UP" in action:
        act_tuple[1] = +1
    actuples.append(tuple(act_tuple))

n = len(transitions)
selected = random.sample(range(0, n), 5000)
st = [ramst[i] for i in selected]
at = [actions[i] for i in selected]
att = [actuples[i] for i in selected]
rt = [rewards[i] for i in selected]
nst = [ramst[i+1] for i in selected]



import ipdb; ipdb.set_trace()