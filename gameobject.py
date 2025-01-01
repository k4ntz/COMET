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

    def __repr__(self):
        return f"{self.name}"