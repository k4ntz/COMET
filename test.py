from worldmodel import WorldModel

wm = WorldModel("Pong")

wm.load_transitions()
wm.add_object("Player", rgb=(92, 186, 92), miny=30)
wm.add_object("Ball", rgb=(236, 236, 236), miny=34, maxy=194)
wm.add_object("Enemy", rgb=(213, 130, 74), miny=30)

for obj in wm.objects:
    wm.detect_objects(obj)
    wm.find_ram(obj.name)
    break

player = wm.objects[0]

import ipdb; ipdb.set_trace()