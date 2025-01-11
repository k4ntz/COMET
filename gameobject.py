import os 
from pyvis.network import Network
import re
import json

COLORS = {
    "blue": '#4bc9dd',
    "red": '#dd4b39',
    "green": '#44aa44',
    "yellow": '#dddd44',
    "white": '#ffffff'
    }

RAM_PATTERN = r'ram\[(\d{1,3})\]'
ACT_PATTERN = r'act\[(\d{1,3})\]'
MIN_PATTERN = r'min\[(\d{1,3})\]'

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

    def make_graph(self):
        self.net = Network(notebook=True, directed=True)
        # self.net.show_buttons(filter_=['physics'])
        self.net.add_node(self.name, label=f'{self.name}', color=COLORS["red"], level = 0)
        # Configure the hierarchical layout
        self.set_net_options()
        rams = self.draw_properties()
        level = 3
        while rams:
            rams = self.draw_connected_rams(rams, level)
            level += 1
        os.makedirs("graphs", exist_ok=True)
        self.net.show(f'graphs/{self.name}.html')

    def draw_properties(self):
        rams = []
        for prop in self.properties:
            if self.equations[prop] is not None:
                self.net.add_node(prop, label=f'{prop}', color=COLORS["green"], 
                                  shape="box", level=1)
                self.net.add_edge(prop, self.name)
                if "ram" in self.equations[prop]:
                    ram_match = re.search(RAM_PATTERN, self.equations[prop]).group(0)
                    self._add_ram_node(ram_match, prop, 2)
                    rams.append(ram_match)
                else:
                    self._add_cst_node(prop, 2)
        return rams

    def _add_ram_node(self, ram_match, prop, level):
        self.net.add_node(ram_match, label=ram_match, shape='box',
                          color=COLORS["blue"], level=level)
        self.net.add_edge(ram_match, prop, title=self.equations[prop])
    
    def _add_cst_node(self, prop, level):
        self.net.add_node(self.equations[prop], label=f'{self.equations[prop]}', 
        shape='box', color=COLORS["white"], level=level)
        self.net.add_edge(self.equations[prop], prop)

                
    def draw_connected_rams(self, rams, level):
        new_rams = []
        # import ipdb; ipdb.set_trace()
        for prop in rams:
            if not isinstance(self.equations[prop], str): # fixing floats
                self.equations[prop] = str(self.equations[prop])
            if "ram" in self.equations[prop]:
                ram_match = re.search(RAM_PATTERN, self.equations[prop]).group(0)
                if ram_match != prop: # avoid cycles
                    self._add_ram_node(ram_match, prop, level)
            if "act" in self.equations[prop]:
                action = re.search(ACT_PATTERN, self.equations[prop]).group(0)
                self.net.add_node(action, label=action,  
                                    shape="box", color=COLORS["yellow"], level=level)
                self.net.add_edge(action, prop, title=self.equations[prop])
            else:
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