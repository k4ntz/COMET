from worldmodel import WorldModel

# wm = WorldModel(game="Pong")
wm = WorldModel(game="PongNoFrameskip-v4")
wm.load_transitions(sample_k=100)
# wm.load_objects()
#wm.update_object_bounds()
#wm.detect_objects()
# wm.find_connected_rams()
# wm.find_all_transitions()
wm.validate_transitions()
# wm.make_graph()

# interventions: test if transitions are correct

# after find_all_transitions:
# wm.ram_equations contains all regressed transitions (equations)
# (e.g. ram[4] = ram[22] + 11)
# iterate over all equations:
#   if equation has other ram location in it:
#       - run game for a bit
#       - at each step: apply intervention (e.g. do(ram[22] = 0))
#       - check if equation still holds (e.g. ram[4] == 11)
#       - if not: remove equation from wm.ram_equations (+print)