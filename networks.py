import random
import operator as op
from functools import reduce
from time import time
import numpy as np
import matplotlib.pyplot as plt
from nodes import *
import itertools
def combination(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

#################################
#                               #
#            Networks           #
#                               #
#################################

class Network:

    """A Network has a list of nodes, and a dictionary of dictionaries representing edges in the
    network. The dictionaries are indexed by nodes, and nodes with any outgoing links are entered
    into the outer dictionary. The inner dictionary maps all nodes with a link from the given node
    to the strength of the link (or 1 if the nodes are unweighted). A Network also has booleans
    directed and weighted, representing whether the Network is directed or weighted, respectively."""

    #################################
    #                               #
    #          Basic Methods        #
    #                               #
    #################################

    def __init__(self, *nodes):
        """Initializes network given list of nodes. Tests all pairs of nodes to determine if
        pairs have a link between them, and documents link in a dictionary indexed by nodes."""
        assert all([type(node) == type(nodes[0]) for node in nodes])
        self.directed = nodes[0].directed
        self.weighted = nodes[0].weighted
        self.nodes = nodes
        self.edges = {}
        for node in self.nodes:
            self.add_node(node)

    def add_node(self, node):
        """Adds a node to the network by updating the Network's list of nodes, the Network's
        ditionary of edges, and the node's network field."""
        if node in self.nodes:
            return
        self.nodes.append(node)
        node.network = self
        if len(node.edges.keys()) > 0:
            self.edges[node] = {}
        for other in self.nodes:
            if other in node.edges:
                self.edges[node][other] = node.edges[other]

    def remove_node(self, node):
        """Removes a node from the network by updating the Network's list of nodes, the Network's
        ditionary of edges, and the node's network field."""
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        node.network = None
        if node in self.edges:
            del self.edges[node]
        for other in self.edges:
            if node in self.edges[other]:
                del self.edges[node][other]

    def get_index(self, node):
        """Returns the index of the argument node contained in the Network, or -1 if the node
        is not in the network."""
        for n in range(len(self.nodes)):
            if node.is_node(self.nodes[n]):
                return n
        return -1

    def has_edge(self, node1, node2):
        """Returns boolean representing whether network contains a link from node1 to node2."""
        return node1 in self.edges and node2 in self.edges[node1] and self.edges[node1][node2] > 0

    def __str__(self):
        """Returns the string version of the dictionary of dictionaries representing the edges."""
        return str(self.edges)

    def __repr__(self):
        """Returns the string version of the list of nodes."""
        return str(self.nodes)

    def copy(self):
        """Returns a copied version of self."""
        copied_nodes = [node.copy() for node in self.nodes]
        for i in range(self.nodes):
            for j in range(self.nodes):
                if self.has_edge(self.nodes[i],self.nodes[j]):
                    copied_nodes[i].add_edge(copied_nodes[j])
        return Network(tuple(copied_nodes))

    #################################
    #                               #
    #      Network Reductions       #
    #                               #
    #################################

    def unweighted_version(self):
        """Returns an unweighted version of the network obtained by normalizing all weights to 1."""
        if not self.weighted:
            return self.copy()
        if self.directed:
            new_nodes = [UDNode(self.nodes[i].name) for i in range(len(self.nodes))]
        else:
            new_nodes = [UUNode(self.nodes[i].name) for i in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.nodes[j] in self.nodes[i].edges:
                    new_nodes[i].add_edge(new_nodes[j])
        return Network(tuple(new_nodes))

    def undirected_version(self):
        """Returns an undirected version of the network obtained by
        1) taking each undirected link between nodes to be the maximum of the magnitude of the previous directed
            links if the Network is weighted,
        2) or simply adding links from j to i whenever a link from i to j exists if the network is unweighted"""
        if not self.directed:
            return self.copy()
        if self.weighted:
            new_nodes = [WUNode(self.nodes[i].name) for i in range(len(self.nodes))]
        else:
            new_nodes = [UUNode(self.nodes[i].name) for i in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.nodes[j] in self.nodes[i].edges:
                    if self.weighted:
                        new_nodes[i].add_edge(new_nodes[j], max(self.nodes[i].edges[self.nodes[j]], float('-inf') if \
                                        self.nodes[i] not in self.nodes[j].edges else self.nodes[j].edges[self.nodes[i]]))
                    else:
                        new_nodes[i].add_edge(new_nodes[j])
        return Network(tuple(new_nodes))

    #################################
    #                               #
    #       Network Matrices        #
    #                               #
    #################################

    def get_matrix(self):
        """Returns the adjacency matrix of the network, in the form of a Numpy array."""
        result = [[0 for _ in len(self.nodes)] for _ in len(self.nodes)]
        for r in range(len(self.nodes)):
            for c in range(len(self.nodes)):
                result[r][c] = 0 if (self.nodes[c] not in self.edges[self.nodes[r]]) else \
                                                    self.edges[self.nodes[r]][self.nodes[c]]
        return np.array(result)

    @staticmethod
    def get_network_from_matrix(matrix):
        """Static method that takes a Numpy array and returns a Network whose adjacency
        matrix is the given array."""
        assert len(matrix.shape()) == 2 and matrix.shape()[0] == matrix.shape()[1]
        for r in matrix.shape()[0]:
            for c in matrix.shape()[0]:
                weighted = (matrix[r][c] != 1 and matrix[r][c] != 0)
                directed = (matrix[r][c] != matrix[c][r])
        if weighted and directed:
            nodes = [WDNode()]*matrix.shape()[0]
        elif weighted and not directed:
            nodes = [WUNode()]*matrix.shape()[0]
        elif directed and not weighted:
            nodes = [UDNode()]*matrix.shape()[0]
        else:
            nodes = [UUNode()]*matrix.shape()[0]
        for c in range(self.cols):
            for r in range(self.rows):
                if self.get_value(r,c) > 0:
                    if directed:
                        WDNode.add_edge(nodes[c], nodes[r], matrix[r][c])
                    else:
                        WUNode.add_edge(nodes[c], nodes[r], matrix[r][c])
        return Network(tuple(nodes))

    #################################
    #                               #
    #         Boolean Network       #
    #           Properties          #
    #                               #
    #################################

    def is_connected(self):
        """Returns True if the network is connected, and false otherwise."""
        return self.make_undirected_version().is_strongly_connected()

    def is_strongly_connected(self):
        """Returns True if the network is strongly connected, and false otherwise."""
        network = self.unweighted_version()
        for i in range(len(network.nodes)):
            for j in range(i,len(network.nodes)):
                if not self.shortest_paths(network.nodes[i], network.nodes[j]):
                    return False
        return True

    def is_bipartite(self):
        """Returns True if the network is bipartite, and false otherwise."""
        return self.bipartite_subsets()[0] is None and self.bipartite_subsets()[1] is None

    def is_empty(self):
        """Returns True if the network is empty, and false otherwise."""
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1 in self.edges and node2 in self.edges[node1]:
                    return False
        return True

    #################################
    #                               #
    #         Scalar Network        #
    #           Properties          #
    #                               #
    #################################

    def total_links(self):
        """Returns the total number of links in the network."""
        total = 0
        for node1 in self.edges:
            for node2 in self.edges[node1]:
                total += 1
        return total if self.directed else total//2

    def sum_links(self):
        """Returns the sum of link magnitudes in the network."""
        total = 0
        for node1 in self.edges:
            for node2 in self.edges[node1]:
                total += self.edges[node1][node2]
        return total if self.directed else total//2

    def density(self):
        """Calculates the density of the network as a ratio of the number of existing
        links to the number of possible links"""
        return (1 if self.directed else 2)*self.total_links()/(len(self.nodes)*(len(self.nodes)-1))

    #################################
    #                               #
    #           Set Network         #
    #           Properties          #
    #                               #
    #################################

    def bipartite_subsets(self):
        """Returns a set of tuples, each tuple containing a bipartite partition of the nodes."""
        bipartite = set()
        if self.is_empty():
            return None, None
        s = set(self.nodes)
        for k in range(1,len(self.nodes)//2 + 1):
            for subset in itertools.combinations(s,k):
                if Network(tuple(subset)).is_empty() and Network(tuple(set(self.nodes) - set(subset))).is_empty():
                    bipartite.add(tuple(subset, set(self.nodes) - set(subset)))
        return bipartite

    def independent_sets(self):
        isets = set()
        for k in range(2,self.nodes):
            for subset in itertools.combinations(s,k):
                if Network(list(subset)).is_empty():
                    isets.add(subset)
        return isets

    def cliques(self):
        """Returns a list of lists, each list representing a clique in the network."""
        cliques = []
        network = self.make_unweighted_version().make_undirected_version()
        starting_nodes = network.nodes.copy()
        while starting_nodes:
            clique = [starting_nodes.pop(0)]
             for node in network.nodes.copy():
                if all([(network.has_edge(clique_node, node) or network.has_edge(node, clique_node)) for clique_node in clique]):
                    clique.append(node)
                    if node in starting_nodes:
                        starting_nodes.remove(node)
            if len(clique) >= 3:
                cliques.append(clique)
        return cliques

    def components(self):
        """Returns a list of lists, each list representing a component of the network."""
        components = []
        network = self.make_unweighted_version().make_undirected_version()
        starting_nodes = network.nodes.copy()
        while starting_nodes:
            component = [starting_nodes.pop(0)]
            for node in starting_nodes.copy():
                if len(network.shortest_paths(node, component[0]))>0 or len(network.shortest_paths(component[0],node))>0:
                    component.append(node)
                for element in component:
                    if element in starting_nodes:
                        starting_nodes.remove(element)
            components.append(component)
        return [Network(tuple(component)) for component in components]

    def degree_partition(self, out=True):
        """Returns a list of lists, with the list at index i representing the nodes
        with degree i. In directed networks, option out determines whether degree
        in question is the in-degree or out-degree."""
        degree_partition = []
        for i in range(len(self.nodes)):
            lst = []
            for node in self.nodes:
                if (self.directed and out) and (i == 0 and node not in self.nodes or \
                                    node in self.nodes and len(self.edges[node].keys()) == i):
                    lst.append(node)
                elif not out:
                    count = 0
                    sources = self.nodes.copy()
                    sources.remove(node)
                    for source in sources:
                        if self.has_edge(source, node):
                            count += 1
                    if count == i:
                        lst.append(node)
            degree_partition.append(lst)
        return degree_partition

    #################################
    #                               #
    #      Distribution Network     #
    #           Properties          #
    #                               #
    #################################

    def degree_count(self, out=True):
        """Returns a list of integers, with integer at index i representing the number
        of nodes with degree i. In directed networks, option out determines whether degree
        in question is the in-degree or out-degree."""
        degree_count = []
        for k in range(len(self.nodes)*(len(self.nodes)-1)//2*(2 if self.directed else 1)):
            degree_count.append(0)
        for node1 in self.edges:
            count = 0
            if out:
                for node2 in self.edges[node1]:
                    count += 1
            else:
                for node2 in self.edges:
                    if self.has_edge(node2, node1):
                        count += 1
            degree_count[count] += 1
        return degree_count

    def degree_dist(self, out=True):
        """Returns a list of floats, with number at index i representing the frequency
        of nodes with degree i. In directed networks, option out determines whether degree
        in question is the in-degree or out-degree."""
        degree_dist = []
        degree_count = self.degree_count()
        for k in range(len(self.nodes)*(len(self.nodes)-1)//2*(2 if self.directed else 1)):
            degree_dist.append(degree_count[k] / len(self.nodes))
        return degree_dist

    #################################
    #                               #
    #        Shortest Paths         #
    #         & Centrality          #
    #                               #
    #################################

    def shortest_paths(self, node1, node2):
        """Returns a list of lists. Each list represents a shortest path between node1 and node2."""
        minsteps = 0
        for n1 in self.edges:
            for n2 in self.edges[n1]:
                minsteps += self.edges[n1][n2]
        lengths = {node1: [([node1],0)]}
        adding = True
        while adding:
            adding = False
            for node in lengths.copy():
                for prevpath in lengths[node]:
                    for neighbor in self.edges[node]:
                        if node is not node2 and prevpath[1] + self.edges[node][neighbor] <= minsteps:
                            if neighbor not in lengths or lengths[neighbor][0][1] > prevpath[1] + self.edges[node][neighbor]:
                                lengths[neighbor] = [(prevpath[0] + [neighbor], prevpath[1] + self.edges[node][neighbor])]
                                adding = True
                            elif lengths[neighbor][0][1] == prevpath[1] + self.edges[node][neighbor] \
                                    and (prevpath[0] + [neighbor], prevpath[1] + self.edges[node][neighbor]) not in lengths[neighbor]:
                                lengths[neighbor].append((prevpath[0] + [neighbor], prevpath[1] + self.edges[node][neighbor]))
                                adding = True
            if node2 in lengths and lengths[node2][0][1] < minsteps:
                minsteps = lengths[node2][0][1]
        if node2 in lengths:
            return [lengths[node2][x][0] for x in range(len(lengths[node2]))]
        return []

    def betweenness(self, node):
        """Returns the betweenness centrality of the node in this network."""
        sum = 0
        for n1 in range(len(self.nodes)):
            for n2 in range(n1+1,len(self.nodes)):
                lst = self.shortest_paths(self.nodes[n1], self.nodes[n2])
                if lst and self.nodes[n1] is not node and self.nodes[n2] is not node:
                    for path in lst.copy():
                        if node not in path:
                            lst.remove(path)
                    sum += len(lst)/len(self.shortest_paths(self.nodes[n1], self.nodes[n2]))
        if len(self.nodes) <= 2:
            return 0
        return 2 * sum / ((len(self.nodes) - 1) * (len(self.nodes) - 2))



#################################
#                               #
#       Network Subclasses      #
#                               #
#################################

class PoissonNetwork(Network):

    """A PoissonNetwork is a Network in which links between nodes are formed randomly.
    PoissonNetworks have a probability of link formation, in addition to the variables
    afforded to them by their Network status."""

    def __init__(self, n, p, **kwargs):
        """Initializes undirected Poisson Network with number of nodes n and probability
        of link formation p. These nodes are named by convention [node1, node2 ... node(n+1)]
        unless a list of names is provided by typing 'names = [list of names]' in the
        initializer. The PoissionNetwork is undirected unless 'directed = True' is called
        in the initializer."""
        assert "names" not in kwargs or len(kwargs["names"]) == n
        self.probability = p
        nodes = []
        for i in range(n):
            name = "node{}".format(i+1) if "names" not in kwargs else kwargs["names"][i]
            if "directed" in kwargs and kwargs["directed"]:
                nodes.append(UDNode(name))
            else:
                nodes.append(UUNode(name))
        for i in range(n):
            if "directed" in kwargs and kwargs["directed"]:
                lst = list(range(i))+list(range(i+1,n))
            else:
                lst = list(range(i+1,n))
            for j in lst:
                if random.random() < p:
                    nodes[i].add_edge(nodes[j])
        Network.__init__(self, nodes)

    def pred_total_link_dist(self):
        """Returns a list, where the element with index i is the theoretical probability
        that the network formed has i total links. When many PoissonNetworks are instantiated
        with this network's size and probability, the distribution of total links should
        match this predicted total link distribution."""
        dist = []
        for m in range(len(self.nodes)*(len(self.nodes)-1)//2*(2 if self.directed else 1)):
            dist.append(pow(self.probability,m)*pow(1-self.probability,len(self.nodes)*(len(self.nodes)-1)//2*(2 if self.directed else 1)-m)\
                    *combination(len(self.nodes)*(len(self.nodes)-1)//2*(2 if self.directed else 1),m))
        return dist

    def pred_degree_dist(self):
        """Returns a list, where the element with index i is the theoretical probability
        that any node has degree i. When a PoissonNetwork is large, it's degree distribution
        should closely match this prediction. When many PoissionNetworks are instantiated and
        each PossionNetwork is large, the degree distribution should match extremely well."""
        dist = []
        dist = []
        for d in range(len(self.nodes)):
            dist.append(combination(len(self.nodes)-1, d) * pow(self.probability, d) * pow(1-self.probability, len(self.nodes)-1-d))
        return dist



class ScaleFreeNetwork(Network):

    """A Network whose nodes have degree distributions following a scale-free distribution.
    Has a gamma attribute that represents the steepness of the exponential decay of the
    degree distribution, and a scalar that normalizes the distribution, as well as all of
    the attributes of a Network."""

    def __init__(self, n, gamma, **kwargs):
        """Initializes a ScaleFreeNetwork based on the number of nodes n and the scale factor
        gamma. If the directed option is included and listed as true, instantiates a randomly
        linked network with out-degree distribution matching the scale-free specifications.
        Otherwise, instantiates a randomly linked network with degree distribution matching the
        scale-free specifications."""
        self.gamma = gamma
        self.scalar = pow(sum([pow(d,-self.gamma) for d in range(1,n)]),-1)
        assert "names" not in kwargs or len(kwargs["names"]) == n
        nodes = []
        for i in range(n):
            name = "node{}".format(i+1) if "names" not in kwargs else kwargs["names"][i]
            if "directed" in kwargs and kwargs["directed"]:
                nodes.append(UDNode(name))
            else:
                nodes.append(UUNode(name))
        i = 0
        if "directed" in kwargs and kwargs["directed"]:
            for d in reversed(range(1,n)):
                for _ in range(round(self.scalar * pow(d, -self.gamma))):
                    for _ in range(d):
                        linkable_nodes = nodes.copy()
                        linkable_nodes.remove(nodes[i])
                        nodes[i].add_edge(random.choice(linkable_nodes))
                        linkeable_nodes.remove(nodes[i])
                    i = i+1
        else:
            assigned_degrees = [1]*n
            j = 1
            while i < n:
                assigned_degrees[i:i+round(self.scalar * pow(j, -self.gamma))] = \
                                        [j]*round(self.scalar * pow(j, -self.gamma))
                i, j = i+round(self.scalar * pow(j, -self.gamma)), j+1
            linkable_indices = range(n)
            j = 0
            while j < n:
                while len(nodes[j].edges.keys()) < assigned_degrees[j]:
                    index_to_link = random.choice(linkable_indices)
                    nodes[j].add_edge(nodes[index_to_link])
                    if len(nodes[index_to_link].edges.keys()) == assigned_degrees[index_to_link]:
                        linkable_indices.remove(index_to_link)
                linkable_indices.remove(j)
                j = j+1
        Network(tuple(nodes))

    def pred_degree_dist(self):
        """Returns a predicted degree-distribution (which should match the
        degree distrbution for large networks) of a scale-free network with
        the given parameters."""
        dist = []
        for d in range(1,len(self.nodes)):
            dist.append(self.scalar * pow(d, -self.gamma))
        return dist
