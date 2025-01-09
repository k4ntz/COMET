import os 
from pyvis.network import Network
import re


COLORS = {
    "blue": '#4bc9dd',
    "red": '#dd4b39',
    "green": '#44aa44',
    }


class GameObject():
    instances = {}

    def __init__(self, name, rgb, minx, maxx, miny, maxy):
        self.name = name
        self.transitions = []
        self.rgb = rgb
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        if name in self.instances:
            self.instances[name] += 1
        else:
            self.instances[name] = 1
        self.x, self.y, self.w, self.h = None, None, None, None
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

    def make_graph(self):
        self.net = Network(notebook=True, directed=True)
        self.net.add_node(self.name, label=f'{self.name}', color=COLORS["red"], x = 0)
        for prop in self.equations:
            if self.equations[prop] is not None:
                self.net.add_node(prop, label=f'{prop}', color=COLORS["green"], x=10)
                if prop in self.properties: # if property, add edge to object
                    self.net.add_edge(self.name, prop) 
                if not isinstance(self.equations[prop], str):
                    self.equations[prop] = str(self.equations[prop])
                if "ram" in self.equations[prop]:
                    ram_pattern = r'ram_(\d{1,3})'
                    ram_match = re.search(ram_pattern, self.equations[prop]).group(0)
                    self.net.add_node(ram_match, label=ram_match, 
                    color=COLORS["blue"], x=20)
                    self.net.add_edge(prop, ram_match, label=self.equations[prop])
                else:
                    self.net.add_node(self.equations[prop], label=f'{self.equations[prop]}', 
                    color=COLORS["blue"])
                    self.net.add_edge(prop, self.equations[prop])
        
        os.makedirs("graphs", exist_ok=True)
        self.net.show(f'graphs/{self.name}.html')

    @property
    def connected_rams(self):
        rams = []
        for prop in self.properties:
            if self.equations[prop] is not None:
                if "ram" in self.equations[prop]:
                    ram_pattern = r'ram_(\d{1,3})'
                    ram_idx = re.search(ram_pattern, self.equations[prop]).group(1)
                    if not ram_idx in rams:
                        rams.append(ram_idx)
        return rams        