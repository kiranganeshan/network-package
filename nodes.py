import random
import operator as op
from functools import reduce
from time import time
import numpy as np
import matplotlib.pyplot as plt
from networks import *


class WDNode:

    """A weighted and directed node. Has class attributes weighted and directed,
    which are set to True, as well as the following instance attributes: a
    dictionary mapping other nodes to the magnitude of links connecting these
    nodes, a network to which the node belongs, and a name."""

    weighted = True
    directed = True

    def __init__(self, name, nodes={}, network=None):
        """Initializes a node based on name, existing dictionary of links,
        and network to which it belongs."""
        self.edges = {}
        self.network = network
        for node in nodes:
            WDNode.add_edge(self, node, nodes[node])
        self.name = name

    def add_edge(self, node, length=1):
        """Adds an edge to another node to this node's dictionary."""
        if node not in self.edges:
            self.edges[node] = length
        if not self.directed and self not in node.edges:
            node.edges[self] = length
        if self.network is not None:
            self.network.edges[self][node] = length
            if not self.directed and self not in node.edges:
                self.network.edges[node][self] = length

    def add_edges(self, *unweighted_nodes, **weighted_nodes):
        """Add many edges based on a dictionary passed in. Format:
        self.add_edges(node1=length1, node2=length2, ...). Also allows
        nodes to be passed in without lengths, in which case lengths
        are set to 1 by default."""
        for node in unweighted_nodes:
            self.add_edge(self, node)
        for node, length in weighted_nodes:
            self.add_edge(self, node, length)

    def __str__(self):
        """A string representation of the node that lists the other nodes
        to which this node is connected."""
        return "{0}: {1}".format(self.name, str(self.edges.keys() if weighted else self.edges))

    def __repr__(self):
        """Simplest string representation of this node: it's name."""
        return self.name

    def copy(self):
        """Returns a copy of this node."""
        if self.directed and self.weighted:
            return WDNode(self.name)
        if self.weighted:
            return WUNode(self.name)
        if self.directed:
            return UDNode(self.name)
        return UUNode(self.name)


class WUNode(WDNode):

    """Same as the WDNode, but not directed. All methods in WDNode account for
    undirected cases, so we need only change the class variable."""

    directed = False



class UDNode(WDNode):

    """A WDNode but unweighted. Most methods stay the same, with minor changes
    to constructor account for the fact that we can now simply take in a list
    of nodes rather than a dictionary."""

    weighted = False

    def __init__(self, name, *nodes, network=None):
        """Transforms list-of-nodes constructor form for unweighted nodes to
        the dictionary-of-nodes constructor form used by WDNode."""
        dict = {}
        for node in nodes:
            dict[node] = 1
        WDNode.__init__(self, name, dict, network)



class UUNode(UDNode):

    """A UDNode, but undirected. Changing only the class variable makes all
    necessary changes."""

    directed = False
