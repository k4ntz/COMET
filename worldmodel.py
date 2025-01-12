import random
import warnings
import pickle as pkl
import numpy as np
import re
import os
from pyvis.network import Network
from PIL import Image

from ocatari.core import OCAtari
from ocatari.vision.utils import find_objects
from ocatari.ram.game_objects import NoObject, ValueObject

from gameobject import GameObject
from utils import remove_constant_and_equivalent, get_model, \
                  replace_vnames, eq_name, RAM_PATTERN

warnings.filterwarnings("ignore")

class WorldModel():
    def __init__(self, game):
        self.game = game
        self.oc_env = OCAtari(game, mode="ram", hud=True, render_mode="rgb_array")

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
        objects = []
        for obj_name, slot_idx in self.slots.items():
            obj = self.oc_env._slots[slot_idx]
            hv = isinstance(obj, ValueObject)
            objects.append(GameObject(obj_name, obj.rgb, value_object=hv))
        self.objects = objects

        self.ram_equations = {}
        self.update_conditions = {}

        self._background_rgb = None
        self._patches_done = False

        np.random.seed(0)
        random.seed(0)

    def load_transitions(self, sample_k=None):
        """
        Load (a subset of) transitions of the game.
        """
        buffer = pkl.load(open(f'transitions/{self.game}.pkl', 'rb'))
        objs, rams, rgbs, actions, rewards, terms, truncs = buffer
        n = len(rams)
        rams = self.convert_rams(rams)
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
        self.rams_mapping, self.equivalents = remove_constant_and_equivalent(self.rams)
        self.acts_mapping, _ = remove_constant_and_equivalent(self.actions)
        
        if not self._background_rgb:
            self._get_background_color()
        if not self._patches_done:
            self._get_objects_patches()

    @property
    def clean_rams(self):
        return self.rams[:, self.rams_mapping]
    
    @property
    def clean_actions(self):
        return self.rams[:, self.acts_mapping]
    
    def unload_transitions(self):
        """
        Unload transitions.
        """
        self.objs, self.rams, self.next_rams, self.rgbs, \
            self.actions, self.rewards, self.terms, self.truncs = None, None, None, None, None, None, None, None
    
    def load_objects(self):
        """
        Load objects from the transitions.
        """
        for i, state in enumerate(self.objs):
            for j, obj in enumerate(self.objects):
                obs_obj = state[j]
                visible = bool(obs_obj)
                obj.visibles.append(visible)
                if visible:
                    if obj.value_object:
                        obj.values.append(obs_obj.value)
                    else:
                        x, y, w, h = obs_obj.xywh
                        obj.xs.append(x)
                        obj.ys.append(y)
                        obj.ws.append(w)
                        obj.hs.append(h)
                else:   
                    if obj.value_object:
                        obj.values.append(None)
                    else:
                        obj.xs.append(None)
                        obj.ys.append(None)
                        obj.ws.append(None)
                        obj.hs.append(None)


    def update_object_bounds(self, name, minx=0, maxx=160, miny=0, maxy=210):
        """
        Update bounds for vision search in image.
        """
        obj_idx = self.slots[name]
        self.objects[obj_idx].update_bounds(minx, maxx, miny, maxy)

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
                # raise ValueError("More than one object detected.")
                obj.visibles.append(False)
                obj.xs.append(None)
                obj.ys.append(None)
                obj.ws.append(None)
                obj.hs.append(None)

    def _split_updated_rams(self, to_track):
        nstates, _ = self.rams.shape
        is_updated_at_state = np.zeros(nstates, dtype=np.int16)
        for i, (state, next_state) in enumerate(zip(self.rams, self.next_rams)):
            if state[to_track] != next_state[to_track]:
                is_updated_at_state[i] = 1
        return is_updated_at_state.astype(np.bool_)

    def regress(self, rams, objective, vnames, prop, obj):
        """
        Perform regression of an object's property on non constant ram states.
        """
        if np.all(objective[:] == objective[0]):
            print(f"\nProperty {prop} of object {obj.name} found constant with value {objective[0]}.")
            # self.objects_properties[obj + "_" + prop] = str(objective[0])
            obj.equations[prop] = str(objective[0])
        else:
            print(f"\nRegressing property {prop} of object {obj.name}.")
            min_val = np.min(objective)
            max_val = np.max(objective)
            model = get_model(l1_loss=True, min_val=min_val, max_val=max_val, complexity_of_vars=4)
            model.fit(rams, objective, variable_names=vnames)
            best = model.get_best()
            eq = best['equation']
            print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
            if input() == 'y':
                # self.objects_properties[obj_name + "_" + prop] = eq
                eq = replace_vnames(eq)
                obj.equations[prop] = eq
            else:
                print(model.equations_)
                print("Enter the equation index that you would like to keep: [Enter digit]")
                eq_idx = int(input())
                eq = model.equations_.loc[eq_idx]['equation']
                eq = replace_vnames(eq)
                obj.equations[prop] = eq
            print(f"Storing equation: `{eq}`.")


    def _find_ram(self, obj):
        """
        Find the properties formulae for the given object.
        Constant ram states are filtered out before running the regression.
        """
        visibles = np.array(obj.visibles)
        rams = self.clean_rams[visibles]
        vnames = [f"ram_{i}" for i in self.rams_mapping]

        for prop in obj.properties:
            if prop != "visible":
                objectives = np.array(obj.__getattribute__(f"{prop}s"))[visibles]
                self.regress(rams, objectives, vnames, prop, obj)


    def find_connected_rams(self):
        """
        Find the connected rams for all objects of the environment.
        """
        for obj in self.objects:
            self._find_ram(obj)

    def _find_hidden_state(self, ram_idx, separate_on_upd=False):
        """
        Find how to update a ram cell at next time step.
        """
        objective = self.next_rams[:, ram_idx]
        max_val = np.max(objective) + 1

        if separate_on_upd:
            is_upd_at_state = self._split_updated_rams(ram_idx)
            vnames = [f"ram_{i}" for i in self.rams_mapping]

            print(f"\nRegressing when {ram_idx} is updated.")
            model = get_model(l1_loss=True, mod_max=max_val,
                              binops=["greater", "logical_or", "logical_and", "mod"])
            model.fit(self.clean_rams, is_upd_at_state, variable_names=vnames)
            best = model.get_best()
            eq = best['equation']
            print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
            if input() != 'y':
                print(model.equations_)
                print("Enter the equation index that you would like to keep: [Enter digit]")
                eq_idx = int(input())
                eq = model.equations_.loc[eq_idx]['equation']
            eq = replace_vnames(eq)
            print(f"Storing equation: `{eq}`.")
            self.update_conditions[f"ram[{ram_idx}]"] = eq

            rams = self.clean_rams[is_upd_at_state]
            acts = self.clean_actions[is_upd_at_state]
            objective = objective[is_upd_at_state]
        
        else:
            rams = self.clean_rams
            acts = self.clean_actions

        # recomputing for the updated only
        extended_rams_and_acts = np.concatenate((rams, acts), axis=1)
        extended_vnames = [f"ram_{i}" for i in self.rams_mapping] \
                        + [f"act_{i}" for i in self.acts_mapping]
        print(f"\nRegressing hidden state of ram {ram_idx}.")
        model = get_model(l1_loss=True, mod_max=max_val, binops=["+", "-", "*", "mod"])
        model.fit(extended_rams_and_acts, objective, variable_names=extended_vnames)
        best = model.get_best()
        eq = best['equation']
        eq = replace_vnames(eq)
        print(f"Regression done. Best equation: `{eq}`. Keep it? [y/n]")
        if input() != 'y':
            print(model.equations_)
            print("Enter the equation index that you would like to keep: [Enter digit]")
            eq_idx = int(input())
            eq = model.equations_.loc[eq_idx]['equation']
            eq = replace_vnames(eq)
        return eq

    def find_transitions(self, obj):
        print(f"\nFinding transitions for object {obj.name}.")
        connected_rams = obj.connected_rams
        print(f"Connected rams: {connected_rams}")
        while connected_rams:
            ram_idx = int(connected_rams.pop(0))
            eqname = eq_name(ram_idx, next=False)
            if eqname in self.ram_equations:
                print(f"Ram {ram_idx} already covered.")
            else:
                eq = self._find_hidden_state(ram_idx, separate_on_upd=False)
                neqname = eq_name(ram_idx, next=True)
                _ = self.compute_accuracy(neqname + " == " + eq)
                print("Do you want to run another regression only on updated rams? [y/n]")
                if input() == 'y':
                    eq = self._find_hidden_state(ram_idx, separate_on_upd=True)
                    _ = self.compute_accuracy(neqname + " == " + eq, separate_on_upd=True)
                print(f"Storing equation: `{eq}`.")
                self.ram_equations[eqname] = eq

                connected_next = re.findall(RAM_PATTERN, eq)
                for rams in connected_next:
                    prefix, nb = rams
                    connected_rams.append(nb)

    def find_all_transitions(self):
        for obj in self.objects:
            self.find_transitions(obj)
    
    def make_graph(self):
        network = Network(notebook=True, directed=True, heading=self.game, 
                          bgcolor=f"rgb{self._background_rgb}", height="800px")
        for obj in self.objects:
            obj.make_graph(network)
        self.net = network
        self.net.show(f'graphs/{self.game}.html')
        print(f"Graph saved in graphs/{self.game}.html")


    def compute_accuracy(self, formulae, separate_on_upd=False):
        """
        Compute accuracy of formulae, e.g.
        `nram[49] == ram[49] - sram[58]`
        `nram[14] == ram[14] + act[1]`

        If option separate_on_upd is on, accuracy is computed only on updated rams.
        """
        formulae = formulae.replace("mod", "np.mod")
        formulae = formulae.replace("greater", "np.greater")
        formulae = formulae.replace("square", "np.square")
        formulae = formulae.replace("neg", "np.negative")
        formulae = formulae.replace("min", "np.minimum")
        formulae = formulae.replace("max", "np.maximum")
        # do not delete unused variables in the following,
        # they are used in the formulae evaluation
        if separate_on_upd:
            pattern = r'nram\[(\d{1,3})\]'
            match = re.search(pattern, formulae)
            if match:
                to_track = int(match.group(1))
                is_upd = self._split_updated_rams(to_track)
                ram, nram = self.rams[is_upd].T, self.next_rams[is_upd].T
                # sram, snram = ram.astype(np.int8), nram.astype(np.int8)
                act = self.actions[is_upd].T
                n = len(ram.T)
            else:
                print("No ram index found in formulae")
                return
        else:
            # sram, ram  = self.rams.astype(np.int8).T, self.rams.T
            nram = self.next_rams.T
            ram  = self.rams.T
            act = self.actions.T
            n = len(self.rams)
        count_matches = np.sum(eval(formulae))
        accuracy = count_matches / n * 100
        print(f"Accuracy of formulae `{formulae}`: {accuracy: .2f}%")
        return accuracy

    def __repr__(self):
        ret = f"WorldModel for game {self.game} with objects:"
        for obj in self.objects: 
            ret += f"\n{obj}"
        return ret
    
    def __getstate__(self):
        self.unload_transitions()
        state = self.__dict__.copy()
        state["game"] = self.game
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

    
    def _get_background_color(self):
        """
        Get the background of the game.
        """
        print("Finding background color.")
        idxs = np.random.randint(len(self.rgbs), size=100)
        flattened = self.rgbs[idxs].reshape(-1, 3)

        # Find all unique colors and their counts
        unique_colors, counts = np.unique(flattened, axis=0, return_counts=True)

        # Pick the color with the highest count
        most_common_color = unique_colors[counts.argmax()]

        # Return as a tuple (e.g., (255, 0, 0))
        self._background_rgb = tuple(most_common_color)
        print(f"Background color found: {self._background_rgb}")

    
    def _get_objects_patches(self):
        self._patches_done = True
        os.makedirs(f"patches/{self.game}", exist_ok=True)
        for i, obj in enumerate(self.objects):
            while True:
                t = random.randint(0, len(self.rgbs)-1)
                if self.objs[t][i].visible:
                    x, y, w, h = self.objs[t][i].xywh
                    patch = self.rgbs[t][y:y + h, x:x + w, :]
                    obj._patchsize = max(w, h)
                    obj._patchpath = f"patches/{self.game}/{obj.name}.png"
                    image = Image.fromarray(patch)
                    image.save(obj._patchpath, 
                               format="PNG", compress_level=0)
                    print(f"Patch for object {obj.name} saved in patches/{self.game}/{obj.name}.png")
                    break
    
    def convert_rams(self, rams):
        """
        Check if the columns contain hexa values and transform them to decimal
        """
        if not hasattr(self, "_hexa_converted"):
            self._hexa_converted = []
            self._signed_converted = []
            rams = rams.astype(np.int16)
            for i in range(rams.shape[1]):
                uniq_val = np.unique(rams[:,i])
                if len(uniq_val) > 10 and all(uniq_val[:11] == [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 16]):
                    print(f"Hexa values detected for ram[{i}]. Transforming to decimal.")
                    rams[:, i] = [int(format(x, 'x')) for x in rams[:, i]]
                    self._hexa_converted.append(i)
                    print(uniq_val, " -> ", np.unique(rams[:, i]))
                elif len(uniq_val) > 1 and max(uniq_val) > 245 and all(uniq_val % 246 < 10):
                    self._signed_converted.append(i)
                    print(f"Signed values detected for ram[{i}]. Transforming to signed.")
                    rams[:, i] = [x - 256 if x > 127 else x for x in rams[:, i]]
                    print(uniq_val, " -> ", np.unique(rams[:, i]))
        else:
            if self._hexa_converted:
                print(f"Hexa values already converted, using already stored: {self._hexa_converted}.")
                for i in self._hexa_converted:
                    rams[:, i] = [int(format(x, 'x')) for x in rams[:, i]]
            if self._signed_converted:
                rams = rams.astype(np.int16)
                print(f"Signed values already converted, using already stored: {self._signed_converted}.")
                for i in self._signed_converted:
                    rams[:, i] = [x - 256 if x > 127 else x for x in rams[:, i]]
        return rams