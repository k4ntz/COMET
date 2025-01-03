import random
import warnings
import pickle as pkl
import numpy as np
import pandas

from ocatari.core import OCAtari
from ocatari.vision.utils import find_objects

from gameobject import GameObject
from utils import remove_constant, get_model

warnings.filterwarnings("ignore")

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
        Load transitions of the game.
        """
        buffer = pkl.load(open(f'transitions/{self.game}.pkl', 'rb'))
        self.objs, self.rams, self.rgbs, self.actions, \
            self.rewards, self.terms, self.truncs = buffer
    
    def sample_transitions(self, N):
        """
        Sample a subset of N transitions and save the subset.
        """
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
                # import ipdb; ipdb.set_trace()
                # raise ValueError("More than one object detected.")
                obj.visibles.append(False)
                obj.xs.append(None)
                obj.ys.append(None)
                obj.ws.append(None)
                obj.hs.append(None)

    def track_object(self, name):
        """
        If the object is visible, track its properties on sampled transitions.
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

    def regress(self, rams, objective, vnames, property, obj):
        """
        Perform regression of an object's property on non constant ram states.
        """
        if np.all(objective[:] == objective[0]):
            print(f"\nProperty {property} of object {obj} found constant with value {objective[0]}.")
            self.objects_properties[obj + "_" + property] = str(objective[0])
        else:
            print(f"\nRegressing property {property} of object {obj}.")
            model = get_model()
            model.fit(rams, objective, variable_names=vnames)
            best = model.get_best()
            eq = best['equation']
            # import ipdb; ipdb.set_trace()
            print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
            if input() == 'y':
                self.objects_properties[obj + "_" + property] = eq
            else:
                print(model.equations_)
                print("Enter the equation index that you would like to keep: [Enter digit]")
                idx = int(input())
                eq = model.equations_.loc[idx]['equation']
                self.objects_properties[obj + "_" + property] = eq
            print(f"Storing equation: `{eq}`.")
        
    def find_ram(self, name):
        """
        Find all the properties formulae for the given object.
        Constant ram states are filtered out before running the regression.
        """
        _, rst, _, _, _ = self.sampled_transitions
        xs, ys, ws, hs, is_visible = self.tracked_objects[name]
        visible_states = rst[is_visible]
        nc_rams, states_poses = remove_constant(visible_states)
        import ipdb; ipdb.set_trace()
        vnames = [f"ram_{i}" for i in states_poses]

        self.regress(nc_rams, xs, vnames, "x", name)
        self.regress(nc_rams, ys, vnames, "y", name)
        self.regress(nc_rams, ws, vnames, "w", name)
        self.regress(nc_rams, hs, vnames, "h", name)

    def find_ram_from_rgb(self, name):
        """
        Find the properties formulae for the given object.
        Constant ram states are filtered out before running the regression.
        """
        obj_slot = self.slots[name]
        obj = self.objects[obj_slot]

        visibles = np.array(obj.visibles)
        nvisibles = np.sum(visibles)
        N = 1000
        sample = random.sample(range(0, nvisibles), N)
        rams = self.rams[visibles][sample]
        xs = np.array(obj.xs)[obj.visibles][sample]
        ys = np.array(obj.ys)[obj.visibles][sample]
        ws = np.array(obj.ws)[obj.visibles][sample]
        hs = np.array(obj.hs)[obj.visibles][sample]
        nc_rams, states_poses = remove_constant(rams)
        vnames = [f"ram_{i}" for i in states_poses]

        self.regress(nc_rams, xs, vnames, "x", name)
        self.regress(nc_rams, ys, vnames, "y", name)
        self.regress(nc_rams, ws, vnames, "w", name)
        self.regress(nc_rams, hs, vnames, "h", name)

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