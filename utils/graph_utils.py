import math
import numpy as np
import networkx as nx
import torch
from itertools import combinations
import time
from networkx.utils import py_random_state

def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets
   
@py_random_state(2)
def generate_graph(n, m, seed=None, hub=None, initial_graph=None):
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )
    

    if initial_graph is None or len(initial_graph)==0:
        # Default initial graph : star graph on (m + 1) nodes
        n_list = list(hub)
        while list(hub)[0] in n_list:
            n_list = seed.sample(range(n),2)
        n_list = list(hub) + n_list 
        G = nx.generators.classic.star_graph(n_list)
        repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(
                f"Barabási–Albert initial graph needs between m={m} and n={n} nodes"
            )
        G = initial_graph.copy()
        repeated_nodes = [n for n, d in G.degree() for _ in range(max(3, d))]

    # Start adding the other n - m nodes.
    node_list = list(range(n))
    print(node_list)
    print(repeated_nodes)
    for node in set(repeated_nodes):
        node_list.remove(node)
        
    while len(node_list) != 0:
        node = node_list.pop(0)
        # Now choose a single from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        # Add edges to 1 node from the source.
        G.add_edges_from(zip([node] * m, targets))
        # Add the node to the list 
        repeated_nodes.extend(targets)
        # And the new node "source" to the list.
        repeated_nodes.extend([node] * m)

    return G
