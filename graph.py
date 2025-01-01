# https://pyvis.readthedocs.io/en/latest/tutorial.html#add-nodes-to-the-network
from pyvis.network import Network

net = Network(notebook=True, directed=True)
net.add_nodes([1], label=['Ball.x'], color=['#4bc9dd'])
net.add_nodes([2], label=['ram49'], color=['#dd4b39'])
net.add_edges([(2, 1)])
net.show('causal_model.html')