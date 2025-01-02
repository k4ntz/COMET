import random
import pickle as pkl
import numpy as np

from ocatari.core import OCAtari
from ocatari.ram.game_objects import NoObject
from ocatari.vision.utils import find_objects

from gameobject import GameObject
from utils import get_model

class WorldModel():
    def __init__(self, game):
        self.oc_env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")
        self.sampled_transitions = None
        self.objects = []
        self.tracked_objects = {}
        self.objects_properties = {}

        # build the slots correspondance
        slots = {}
        k = 0
        for cat, nb in self.oc_env.max_objects_per_cat.items():
            slots[cat] = k # [i for i in range(k, k+nb)]
            k += nb
        self.slots = slots

        np.random.seed(0)
        random.seed(0)
    
    @property
    def game(self):
        return self.oc_env.game_name

    def load_transitions(self):
        """
        Load transitions of the game, sample a subset of N transitions and save the subset.
        """
        buffer = pkl.load(open(f'transitions/{self.game}.pkl', 'rb'))
        self.objs, self.rams, self.rgbs, self.actions, \
            self.rewards, self.terms, self.truncs = buffer
    
    def sample_transitions(self, N):
        n = len(self.objs)
        sample = random.sample(range(0, n), N)
        ost = np.array([self.objs[i] for i in sample])
        rst = np.array([self.rams[i] for i in sample])
        at = np.array([self.actions[i] for i in sample])
        rt = np.array([self.rewards[i] for i in sample])
        nrst = np.array([self.rams[i+1] for i in sample])
        self.sampled_transitions = ost, rst, at, rt, nrst
    
    def add_object(self, name, rgb, minx=0, maxx=160, miny=0, maxy=210):
        """
        Add an object to the world model.
        """
        self.objects.append(GameObject(name, rgb, minx, maxx, miny, maxy))

    def detect_objects(self, obj):
        """
        Detect objects in the transitions.
        """
        for rgbst in self.rgbs:
            objs = find_objects(rgbst, obj.rgb, minx=obj.minx, maxx=obj.maxx, miny=obj.miny, maxy=obj.maxy)
            if len(objs) == 0:
                obj.visibles.append(False)
                obj.xs.append(None)
                obj.ys.append(None)
                obj.ws.append(None)
                obj.hs.append(None)
            elif len(objs) == 1:
                x, y, w, h = objs[0]
                obj.visibles.append(True)
                obj.xs.append(x)
                obj.ys.append(y)
                obj.ws.append(w)
                obj.hs.append(h)
            else:
                raise ValueError("More than one object detected.")

    def track_object(self, name):
        """
        If the object is visible, track its properties.
        """
        if self.sampled_transitions == None:
            print("Please load and sample transitions before proceeding.")
            return
        
        ost, _, _, _, _ = self.sampled_transitions

        # filtering invisible objects
        is_visible = np.array([name in [o.category for o in objs] for objs in ost])
        visible_ost = ost[is_visible]
        nstates = len(visible_ost)

        # getting the objectives
        obj_slot = self.slots[name]
        selected_obj = visible_ost[:, obj_slot]
        xs, ys, ws, hs = np.zeros(nstates), np.zeros(nstates), np.zeros(nstates), np.zeros(nstates)
        for j, obj in enumerate(selected_obj):
            x, y, w, h = obj.xywh
            xs[j] = x
            ys[j] = y
            ws[j] = w
            hs[j] = h

        self.tracked_objects[name] = xs, ys, ws, hs, is_visible
        
    def find_ram(self, name):
        # finds the ram of the non constant properties of the object(s) with the given name
        _, rst, _, _, _ = self.transitions
        xs, ys, ws, hs, is_visible = self.tracked_objects[name]
        ram_states = rst[is_visible]

        if np.all(xs[:] == xs[0]):
            self.objects_properties[name + "_x"] = str(xs[0])
        else:
            model = get_model()
            model.fit(ram_states, xs)

        if np.all(ys[:] == ys[0]):
            self.objects_properties[name + "_y"] = str(ys[0])
        else:
            model = get_model()
            model.fit(ram_states, ys)

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

    def __repr__(self):
        ret = f"WorldModel for game {self.game} with objects:"
        for obj in self.objects: 
            ret += f"\n{obj}"
        return ret