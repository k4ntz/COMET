from argparse import ArgumentParser
import os
import pickle as pkl
import random
import numpy as np

np.random.seed(0)
random.seed(0)

parser = ArgumentParser()
parser.add_argument("-g", "--game", type=str, default="Pong")

args = parser.parse_args()

transitions = pkl.load(open(f'transitions/{args.game}.pkl', 'rb'))

objs, rams, rgbs, actions, rewards, terms, truncs = transitions


n = len(objs)
selected = random.sample(range(0, n), 500)
st = np.array([rams[i] for i in selected])
att = np.array([actions[i] for i in selected])
rt = np.array([rewards[i] for i in selected])
nst = np.array([rams[i+1] for i in selected])

import ipdb; ipdb.set_trace()