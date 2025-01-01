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