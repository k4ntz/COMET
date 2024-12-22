import pickle as pkl
import numpy as np
import random


class GameObject():
    instances = {}

    def __init__(self, name, rgb):
        self.name = name
        self.transitions = []
        self.rgb = rgb
        if name in self.instances:
            self.instances[name] += 1
        else:
            self.instances[name] = 1

    def __str__(self):
        return f"{self.name} at ({self.x}, {self.y})"


class Environment():
    def __init__(self, env):
        self.oc_env = env
        self.tracked_objects = {} # the name and number of instance for each class, and their class
        self.classes = {}
        self.transitions = [] # [(st, at, rt, st+1) for _ in range(N)] with st = (ram_t, rgb_t)
        np.random.set_seed(0)
        random.seed(0)

    def load_transitions(self, path):
        # sample a subset of ~N transitions # save the subset
        

    def track_new_object(self, name, color, minx, miny, maxx, maxy):
        # detects the max nb of objects of this class in the image
        # extracts the object(s) from the transitions rgb st and st+1 
        
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

    def load_transitions(self):
        # records a set of transitions
        

    def track_new_object(self, name, color, minx, miny, maxx, maxy):
        # extracts the object(s) from the transitions rgb st and st+1
        # extracts visible and x, y, w, h, rgb from the transitions

        
    def _init_objects_graphs(self, object):
        # creates a graph of the objects
        pass

    def _add_transitions_to_graph(self):
        # adds the transitions to the graph
        pass
    
    @property
    def name(self):
        return self.oc_env.name