import os 
from pyvis.network import Network
import re
import json
import uuid

from utils import replace_float_with_int_if_close, encode_image_to_base64, \
                  RAM_PATTERN, ACT_PATTERN

COLORS = {
    "blue": '#4bc9dd',
    "red": '#dd4b39',
    "green": '#44aa44',
    "yellow": '#dddd44',
    "white": '#ffffff'
    }

class GameObject():
    def __init__(self, name, rgb, minx=0, maxx=160, miny=0, maxy=210, value_object=False):
        self.name = name
        self.rgb = rgb
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.value_object = value_object

        if value_object:
            self.visibles, self.values = [], []
            self.properties = ["visible", "value"]
        else:
            self.xs, self.ys, self.ws, self.hs, \
                self.visibles = [], [], [], [], []
            self.properties = ["x", "y", "w", "h", "visible"]
        
        self.equations = {prop: None for prop in self.properties}

    def __repr__(self):
        string = f"{self.name}"
        for prop in self.properties:
            if self.equations[prop] is not None:
                string += f"\n\t{prop}: {self.equations[prop]}"
        return string

    def update_bounds(self, minx=0, maxx=160, miny=0, maxy=210):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
    
    def clean_equations(self):
        for eq_id in self.equations:
            eq = self.equations[eq_id]
            if eq is not None:
                if not isinstance(eq, str):
                    eq = str(eq)
                self.equations[eq_id] = replace_float_with_int_if_close(eq)


    def make_graph(self, network=None):
        if network is None:
            self.net = Network(notebook=True, directed=True)
        else:
            self.net = network
        # self.net.show_buttons(filter_=['physics'])
        image_path = encode_image_to_base64(self._patchpath)
        self.net.add_node(self.name, label=f'{self.name}', 
                          color=COLORS["red"], level = 0,
                          shape="image",
                          image=image_path,
                          size=int(self._patchsize))
        # Configure the hierarchical layout
        # self.set_net_options()
        self.clean_equations()
        rams = self.draw_properties()
        covered_rams = [ram for ram in rams]
        level = 3
        while rams:
            rams = self.draw_connected_rams(rams, level)
            level += 1
            for ram in rams:
                if ram in covered_rams:
                    rams.remove(ram)
                if not ram in covered_rams:
                    covered_rams.append(ram)
        os.makedirs("graphs", exist_ok=True)
        if network is None:
            self.net.heading = self.name
            self.net.show(f'graphs/{self.name}.html')

    def draw_properties(self):
        rams = []
        for prop in self.properties:
            if self.equations[prop] is not None:
                nid = self.name + '.' + prop
                self.net.add_node(nid, label=f'{prop}', color=COLORS["green"], 
                                  shape="box", level=1)
                self.net.add_edge(nid, self.name)
                if "ram" in self.equations[prop]:
                    ram_match = re.search(RAM_PATTERN, self.equations[prop]).group(0)
                    self.net.add_node(ram_match, label=ram_match, shape='box',
                                      color=COLORS["blue"], level=2)
                    self.net.add_edge(ram_match, nid, title=self.equations[prop])
                    if not ram_match in rams:
                        rams.append(ram_match)
                else:
                    uniq_id = uuid.uuid4().hex
                    self.net.add_node(uniq_id, label=f'{self.equations[prop]}', 
                          shape='box', color=COLORS["white"], level=2)
                    self.net.add_edge(uniq_id, nid)
        return rams

    def _add_ram_node(self, ram_match, prop, level):
        self.net.add_node(ram_match, label=ram_match, shape='box',
                          color=COLORS["blue"], level=level)
        self.net.add_edge(ram_match, prop, title=self.equations[prop])
    
    def _add_cst_node(self, prop, level):
        uniq_id = uuid.uuid4().hex
        self.net.add_node(uniq_id, label=f'{self.equations[prop]}', 
                          shape='box', color=COLORS["white"], level=level)
        self.net.add_edge(uniq_id, prop)

                
    def draw_connected_rams(self, rams, level):
        new_rams = []
        for prop in rams:
            if not isinstance(self.equations[prop], str): # fixing floats
                self.equations[prop] = str(self.equations[prop])
            is_variable = False
            if "ram" in self.equations[prop]:
                # TODO add check for multiple rams in equation
                ram_match = re.search(RAM_PATTERN, self.equations[prop]).group(0)
                if ram_match != prop: # avoid cycles
                    self._add_ram_node(ram_match, prop, level)
                is_variable = True
            if "act" in self.equations[prop]:
                action = re.search(ACT_PATTERN, self.equations[prop]).group(0)
                self.net.add_node(action, label=action,  
                                    shape="box", color=COLORS["yellow"], level=level)
                self.net.add_edge(action, prop, title=self.equations[prop])
                is_variable = True
            if not is_variable:
                self.net.add_node(self.equations[prop], label=f'{self.equations[prop]}', 
                                  color=COLORS["blue"])
                self.net.add_edge(self.equations[prop], prop)
        return new_rams
    
    
    def set_net_options(self):
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",  # Top-Down (use "LR" for left-to-right)
                    "sortMethod": "directed",  # Organize nodes based on direction
                }
            }
        }
        # Set the options in the Pyvis network
        self.net.set_options(json.dumps(options))

    @property
    def connected_rams(self):
        rams = []
        for prop in self.properties:
            if self.equations[prop] is not None:
                if "ram" in self.equations[prop]:
                    ram_idx = re.search(RAM_PATTERN, self.equations[prop]).group(1)
                    if not ram_idx in rams:
                        rams.append(ram_idx)
        return rams
    


