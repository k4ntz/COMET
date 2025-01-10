import random
import warnings
import pickle as pkl
import numpy as np
import pandas

from ocatari.core import OCAtari
from ocatari.vision.utils import find_objects

from gameobject import GameObject
from utils import remove_constant, get_model, split_constant_variable_rams, extend_with_signed_rams

warnings.filterwarnings("ignore")

class WorldModel():
    def __init__(self, game):
        self.oc_env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")
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

    def load_transitions(self, sample_k=None):
        """
        Load (a subset of) transitions of the game.
        """
        buffer = pkl.load(open(f'transitions/{self.game}.pkl', 'rb'))
        objs, rams, rgbs, actions, rewards, terms, truncs = buffer
        n = len(rams)
        if sample_k is None:
            self.sample = None
            self.objs, self.rams, self.next_rams, self.rgbs, \
                self.actions, self.rewards, self.terms, self.truncs \
                    = objs[:-1], rams[:-1], rams[1:], rgbs[:-1], \
                        actions[:-1], rewards[:-1], terms[:-1], truncs[:-1]
        else:
            sample = np.array(random.sample(range(0, n-1), sample_k))
            self.sample = sample
            self.objs, self.rams, self.next_rams, self.rgbs, \
                self.actions, self.rewards, self.terms, self.truncs \
                    = objs[sample], rams[sample], rams[sample+1], rgbs[sample], \
                        actions[sample], rewards[sample], terms[sample], truncs[sample]
    
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

    def regress(self, rams, objective, vnames, property, obj):
        """
        Perform regression of an object's property on non constant ram states.
        """
        if np.all(objective[:] == objective[0]):
            print(f"\nProperty {property} of object {obj} found constant with value {objective[0]}.")
            self.objects_properties[obj + "_" + property] = str(objective[0])
        else:
            print(f"\nRegressing property {property} of object {obj}.")
            min_val = np.min(objective)
            max_val = np.max(objective)
            model = get_model(l1_loss=True, min_val=min_val, max_val=max_val)
            model.fit(rams, objective, variable_names=vnames)
            best = model.get_best()
            eq = best['equation']
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
        Find the properties formulae for the given object.
        Constant ram states are filtered out before running the regression.
        """
        obj_slot = self.slots[name]
        obj = self.objects[obj_slot]

        visibles = np.array(obj.visibles)
        rams = self.rams[visibles]
        xs = np.array(obj.xs)[visibles]
        ys = np.array(obj.ys)[visibles]
        ws = np.array(obj.ws)[visibles]
        hs = np.array(obj.hs)[visibles]
        nc_rams, rams_mapping = remove_constant(rams)
        vnames = [f"ram_{i}" for i in rams_mapping]

        self.regress(nc_rams, xs, vnames, "x", name)
        self.regress(nc_rams, ys, vnames, "y", name)
        self.regress(nc_rams, ws, vnames, "w", name)
        self.regress(nc_rams, hs, vnames, "h", name)


    def find_hidden_state(self, idx):
        """
        Find how to update a ram cell at next time step.
        """
        is_constant_at_state, non_cst_rams, non_cst_next_rams \
            = split_constant_variable_rams(self.rams, self.next_rams, idx)
        non_cst_acts = self.actions[~is_constant_at_state]
        objective = non_cst_next_rams[:, idx]

        nc_rams, rams_mapping = remove_constant(non_cst_rams)
        extended_rams = extend_with_signed_rams(nc_rams)
        nc_acts, acts_mapping = remove_constant(non_cst_acts)
        extended_rams_and_acts = np.concatenate((extended_rams, nc_acts), axis=1)
        extended_vnames = [f"s{i}" for i in rams_mapping] + [f"ss{i}" for i in rams_mapping] \
                         + [f"a{i}" for i in acts_mapping]

        print(f"\nRegressing hidden state of ram {idx}.")
        model = get_model(l1_loss=True, binops=["+", "-", "*", "/"])
        model.fit(extended_rams_and_acts, objective, variable_names=extended_vnames)
        best = model.get_best()
        eq = best['equation']
        print(f"Regression done. Best equation: `{eq}`")

        nc_rams, rams_mapping = remove_constant(self.rams)
        vnames = [f"ram_{i}" for i in rams_mapping]

        print(f"\nRegressing when {idx} is constant.")
        model = get_model(l1_loss=True, binops=["greater", "logical_or", "logical_and", "mod"])
        model.fit(nc_rams, is_constant_at_state, variable_names=vnames)
        best = model.get_best()
        eq = best['equation']
        print(f"Regression done. Best equation: `{eq}`")

    def compute_accuracy(self, formulae):
        """
        Compute accuracy of formulae e.g. `ns[49] == s[49] - ss[58]`.
        """
        formulae = formulae.replace("mod", "np.mod")
        formulae = formulae.replace("greater", "np.greater")
        formulae = formulae.replace("equal", "np.equal")
        formulae = formulae.replace("square", "np.square")
        formulae = formulae.replace("neg", "np.negative")
        formulae = formulae.replace("max", "np.maximum")
        # do not delete the following, used in the formulae evaluation
        sns, ns = self.next_rams.astype(np.int8).T, self.next_rams.T
        ss, s  = self.rams.astype(np.int8).T, self.rams.T
        a = self.actions.T
        count_matches = np.sum(eval(formulae))
        print(f"Accuracy of regression: {count_matches / len(self.rams) * 100: .2f}%")

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