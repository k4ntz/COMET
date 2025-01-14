from worldmodel import WorldModel
from argparse import ArgumentParser
import pickle as pkl
import os

parser = ArgumentParser("test parser")
parser.add_argument("--game", type=str, default="Pong")
parser.add_argument("--from-scratch", action="store_true")

args = parser.parse_args()

if args.from_scratch or not os.path.exists(f"worldmodels/{args.game}.pkl"):
    print("Building world model from scratch for game", args.game)
    wm = WorldModel(args.game)

    wm.load_transitions()
    if False:
        wm.add_object("Player", rgb=(92, 186, 92), miny=30)
        wm.add_object("Ball", rgb=(236, 236, 236), miny=34, maxy=194)
        wm.add_object("Enemy", rgb=(213, 130, 74), miny=30)
        for obj in wm.objects:
            wm.detect_objects(obj)
    else:
        wm.load_objects()
    
    wm.find_connected_rams()
    
    os.makedirs("worldmodels", exist_ok=True)
    pkl.dump(wm, open(f"worldmodels/{args.game}.pkl", "wb"))
    print(f"World model saved in worldmodels/{args.game}.pkl")

else:
    wm = pkl.load(open(f"worldmodels/{args.game}.pkl", "rb"))
    import ipdb; ipdb.set_trace()
    wm.game = wm.oc_env.game_name
    wm.load_transitions()
    # wm._get_objects_patches()
    # pkl.dump(wm, open(f"worldmodels/{args.game}.pkl", "wb"))
    # exit()
    # import ipdb; ipdb.set_trace()

# for obj in wm.objects:
#     wm.find_transitions(obj)
#     print(wm.ram_equations)

# pkl.dump(wm, open(f"worldmodels/{args.game}.pkl", "wb"))

wm.make_graph()

# wm.objects[0].make_graph()

# for obj in wm.objects:
#     obj.make_graph()
    #

# import ipdb; ipdb.set_trace()