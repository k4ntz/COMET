import random
import pickle as pkl
import numpy as np

from ocatari.core import OCAtari
from ocatari.ram.game_objects import NoObject

from .utils import is_constant

class WorldModel():
    def __init__(self, game):
        self.oc_env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")
        self.transitions = None
        self.tracked_objects = {}
        self.objects_properties = {}

        # build the slots correspondance
        slots = {}
        k = 0
        for cat, nb in self.env.max_objects_per_cat.items():
            slots[cat] = k # [i for i in range(k, k+nb)]
            k += nb
        self.slots = slots

        np.random.set_seed(0)
        random.seed(0)
    
    @property
    def game(self):
        return self.oc_env.game_name

    def load_transitions(self, N):
        """
        Load transitions of the game, sample a subset of N transitions and save the subset.
        """
        buffer = pkl.load(open(f'transitions/{self.game}.pkl', 'rb'))
        objs, rams, rgbs, actions, rewards, terms, truncs = buffer
        n = len(objs)
        sample = random.sample(range(0, n), N)
        ost = np.aray([objs[i] for i in sample])
        rst = np.array([rams[i] for i in sample])
        at = np.array([actions[i] for i in sample])
        rt = np.array([rewards[i] for i in sample])
        nrst = np.array([rams[i+1] for i in sample])
        self.transitions = ost, rst, at, rt, nrst
        
    def track_object(self, name):
        """
        If the object is visible, track its properties.
        """
        if self.transitions == None:
            print("Please load transitions before proceeding.")
            return
        
        ost, _, _, _, _ = self.transitions

        # filtering invisible objects
        is_visible = np.array([name in [o.category for o in objs] for objs in ost])
        visible_ost = ost[is_visible]
        nstates = len(visible_ost)

        # getting the objectives
        obj_slot = self.slots[name]
        selected_obj = visible_ost[:, obj_slot]
        xs, ys, ws, hs = np.zeros(nstates)
        for j, obj in enumerate(selected_obj):
            x, y, w, h = obj.xywh
            xs[j] = x
            ys[j] = y
            ws[j] = w
            hs[j] = h

        self.tracked_objects[name] = xs, ys, ws, hs, is_visible
        
    def find_ram(self, name):
        # finds the ram of the non constant properties of the object(s) with the given name
        pass

    def find_transitions(self):
        # finds the transitions that contain the object with the given name
        for obj in self.objects:
            pass # find the visibility criterion and the transitions if object is visible 

    def show_graph(self):
        # shows the graph of the objects
        pass
        
    def _init_objects_graphs(self, object):
        # creates a graph of the objects
        pass

    def _add_transitions_to_graph(self):
        # adds the transitions to the graph
        pass