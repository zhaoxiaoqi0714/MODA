import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random

class community():
	"""community class definition"""
	def __init__(self, nodes, prob):
		self.nodes = nodes
		self.prob = prob

# AGM generates graph from communities, Matrix F is 0/1
def AGM(nodes,communities):
	# new an empty graph
	G = nx.Graph()
	# add nodes into the graph
	G.add_nodes_from(nodes)
	# generate edges
	for c in communities:
		# combinations combines #param2 nodes from #para1 for all the cases
		for pairs in combinations(c.nodes, 2):
			if random.random() <= c.prob:
				G.add_edge(pairs[0], pairs[1])
	return G








