import random
import warnings
import pickle as pkl
import numpy as np
import pandas
import re

from ocatari.core import OCAtari
from ocatari.vision.utils import find_objects
from ocatari.ram.game_objects import NoObject

from gameobject import GameObject
from utils import remove_constant, get_model, split_constant_variable_rams, extend_with_signed_rams

warnings.filterwarnings("ignore")

class WorldModel():
    def __init__(self, game):
        self.oc_env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")
        self.objects = []
        self.update_conditions = {}

        # build the slots correspondance
        slots = {}
        k = 0
        for cat, max_nb in self.oc_env.max_objects_per_cat.items():
            if max_nb == 1:
                slots[cat] = k
                k+=1
            else:
                for i in range(max_nb):
                    slot_name = cat + str(i+1)
                    slots[slot_name] = k
                    k+=1
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
    
    def load_objects(self):
        """
        Load objects from the transitions.
        """
        objects = []
        for obj_name, slot_idx in self.slots.items():
            rgb = self.objs[0][slot_idx].rgb
            obj = GameObject(obj_name, rgb)
            objects.append(obj)

        for i, state in enumerate(self.objs):
            for j, obj in enumerate(objects):
                if type(state[j]) is NoObject:
                    obj.visibles.append(False)
                    obj.xs.append(None)
                    obj.ys.append(None)
                    obj.ws.append(None)
                    obj.hs.append(None)
                else:
                    x, y, w, h = state[j].xywh
                    obj.visibles.append(True)
                    obj.xs.append(x)
                    obj.ys.append(y)
                    obj.ws.append(w)
                    obj.hs.append(h)
        self.objects = objects


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

    def regress(self, rams, objective, vnames, prop, obj_name):
        """
        Perform regression of an object's property on non constant ram states.
        """
        obj_slot = self.slots[obj_name]
        obj = self.objects[obj_slot]
        if np.all(objective[:] == objective[0]):
            print(f"\nProperty {prop} of object {obj_name} found constant with value {objective[0]}.")
            # self.objects_properties[obj + "_" + prop] = str(objective[0])
            obj.equations[prop] = str(objective[0])
        else:
            print(f"\nRegressing property {prop} of object {obj_name}.")
            min_val = np.min(objective)
            max_val = np.max(objective)
            model = get_model(l1_loss=True, min_val=min_val, max_val=max_val)
            model.fit(rams, objective, variable_names=vnames)
            best = model.get_best()
            eq = best['equation']
            print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
            if input() == 'y':
                # self.objects_properties[obj_name + "_" + prop] = eq
                eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
                obj.equations[prop] = eq
            else:
                print(model.equations_)
                print("Enter the equation index that you would like to keep: [Enter digit]")
                eq_idx = int(input())
                eq = model.equations_.loc[eq_idx]['equation']
                # self.objects_properties[obj_name + "_" + property] = eq
                eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
                obj.equations[prop] = eq
            print(f"Storing equation: `{eq}`.")


    def _find_ram(self, obj):
        """
        Find the properties formulae for the given object.
        Constant ram states are filtered out before running the regression.
        """
        visibles = np.array(obj.visibles)
        rams = self.rams[visibles]
        nc_rams, rams_mapping = remove_constant(rams)
        vnames = [f"ram_{i}" for i in rams_mapping]

        for prop in obj.properties:
            if prop != "visible":
                objectives = np.array(obj.__getattribute__(f"{prop}s"))[visibles]
                self.regress(nc_rams, objectives, vnames, prop, obj.name)

    def find_connected_rams(self):
        """
        Find the connected rams for all objects of the environment.
        """
        for obj in self.objects:
            self._find_ram(obj)

    def _find_hidden_state(self, ram_idx, separate_on_cst=False):
        """
        Find how to update a ram cell at next time step.
        """
        if separate_on_cst:
            is_constant_at_state, non_cst_rams, non_cst_next_rams \
                = split_constant_variable_rams(self.rams, self.next_rams, ram_idx)

            nc_rams, rams_mapping = remove_constant(self.rams)
            vnames = [f"ram_{i}" for i in rams_mapping]

            print(f"\nRegressing when {ram_idx} is constant.")
            model = get_model(l1_loss=True, binops=["greater", "logical_or", "logical_and", "mod"])
            model.fit(nc_rams, is_constant_at_state, variable_names=vnames)
            best = model.get_best()
            eq = best['equation']
            print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
            if input() != 'y':
                print(model.equations_)
                print("Enter the equation index that you would like to keep: [Enter digit]")
                eq_idx = int(input())
                eq = model.equations_.loc[eq_idx]['equation']
            eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
            print(f"Storing equation: `{eq}`.")
            self.update_conditions[f"ram[{ram_idx}]"] = eq

            rams = non_cst_rams
            acts = self.actions[~is_constant_at_state]
            next_rams = non_cst_next_rams

        else:
            rams = self.rams
            acts = self.actions
            next_rams = self.next_rams

        nc_rams, rams_mapping = remove_constant(rams)
        extended_rams = extend_with_signed_rams(nc_rams)
        nc_acts, acts_mapping = remove_constant(acts)
        extended_rams_and_acts = np.concatenate((extended_rams, nc_acts), axis=1)
        extended_vnames = [f"ram_{i}" for i in rams_mapping] + [f"sram_{i}" for i in rams_mapping] \
                         + [f"act_{i}" for i in acts_mapping]
        print(f"\nRegressing hidden state of ram {ram_idx}.")
        objective = next_rams[:, ram_idx]
        model = get_model(l1_loss=True, binops=["+", "-", "*", "/", "mod"])
        model.fit(extended_rams_and_acts, objective, variable_names=extended_vnames)
        best = model.get_best()
        eq = best['equation']
        eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
        print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
        if input() != 'y':
            print(model.equations_)
            print("Enter the equation index that you would like to keep: [Enter digit]")
            eq_idx = int(input())
            eq = model.equations_.loc[eq_idx]['equation']
            eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
        print(f"Storing equation: `{eq}`.")
        return eq

    def find_transitions(self, obj):
        print(f"\nFinding transitions for object {obj.name}.")
        print(obj.connected_rams)
        for ram_idx in obj.connected_rams:
            eq = self._find_hidden_state(int(ram_idx))
            obj.equations[f"ram[{ram_idx}]"] = eq

    def compute_accuracy(self, formulae, separate_on_cst=False):
        """
        Compute accuracy of formulae, e.g.
        `nram[49] == ram[49] - sram[58]`
        `nram[14] == ram[14] + act[1]`

        If option separate_on_cst is on, accuracy is computed only on updated rams.
        """
        formulae = formulae.replace("mod", "np.mod")
        formulae = formulae.replace("greater", "np.greater")
        formulae = formulae.replace("equal", "np.equal")
        formulae = formulae.replace("square", "np.square")
        formulae = formulae.replace("neg", "np.negative")
        formulae = formulae.replace("max", "np.maximum")
        # do not delete unused variables in the following,
        # they are used in the formulae evaluation
        if separate_on_cst:
            pattern = r'nram\[(\d{1,3})\]'
            match = re.search(pattern, formulae)
            if match:
                to_track = int(match.group(1))
                is_cst, ram, nram = split_constant_variable_rams(self.rams, self.next_rams, to_track)
                # import ipdb; ipdb.set_trace()
                n = len(ram)
                ram, nram = ram.T, nram.T
                sram, snram = ram.astype(np.int8), nram.astype(np.int8)
                act = self.actions[~is_cst].T
            else:
                print("No ram index found in formulae")
                return
        else:
            snram, nram = self.next_rams.astype(np.int8).T, self.next_rams.T
            sram, ram  = self.rams.astype(np.int8).T, self.rams.T
            act = self.actions.T
            n = len(self.rams)
        count_matches = np.sum(eval(formulae))
        accuracy = count_matches / n * 100
        print(f"Accuracy of formulae `{formulae}`: {accuracy: .2f}%")
        return accuracy

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
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["game"] = self.oc_env.game_name
        del state["oc_env"]
        return state
    
    def __setstate__(self, state):
        game = state.pop("game")
        self.__dict__.update(state)
        self.oc_env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")

    @property
    def objects_properties(self):
        dico = {}
        for obj in self.objects:
            for prop in obj.properties:
                dico[obj.name + "_" + prop] = obj.equations[prop]
        return dico
    
    # @objects_properties.setter
    # def objects_properties(self, dico):
    #     pass