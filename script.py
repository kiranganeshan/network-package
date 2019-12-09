import random
import operator as op
from functools import reduce
from time import time
import numpy as np
import matplotlib.pyplot as plt
from nodes import *
from networks import *



def medici():
    """Instantiates the nodes in the medici network example, and calculates
    the cliques and components of the network as well as the Betweenness
    of the Medici family in the network."""
    pucci = UUNode("pucci")
    peruzzi = UUNode("peruzzi")
    bischeri = UUNode("bischeri")
    lambertes = UUNode("lambertes")
    strozzi = UUNode("strozzi")
    guadagni = UUNode("guadagni")
    ridolfi = UUNode("ridolfi")
    tornabuon = UUNode("tornabuon")
    castellan = UUNode("castellan")
    medici = UUNode("medici")
    albizzi = UUNode("albizzi")
    barbadori = UUNode("barbadori")
    acciaiuol = UUNode("acciaiuol")
    salvati = UUNode("salvati")
    ginori = UUNode("ginori")
    pazzi = UUNode("pazzi")
    peruzzi.add_edges(bischeri, strozzi, castellan)
    bischeri.add_edges(strozzi, guadagni)
    lambertes.add_edge(guadagni)
    strozzi.add_edges(castellan, ridolfi)
    guadagni.add_edges(tornabuon, albizzi)
    ridolfi.add_edges(tornabuon, medici)
    tornabuon.add_edge(medici)
    castellan.add_edge(barbadori)
    medici.add_edges(barbadori, albizzi, acciaiuol, salvati)
    albizzi.add_edge(ginori)
    salvati.add_edge(pazzi)
    florentines = Network([pucci, peruzzi, bischeri, lambertes, strozzi, guadagni, ridolfi, tornabuon, \
                castellan, medici, albizzi, barbadori, acciaiuol, salvati, ginori, pazzi])
    print("Florentine Cliques: ", florentines.cliques())
    print("Florentine Components: ", florentines.components())
    print("Betweenness of the Medici: ", florentines.betweenness(medici))



def poisson(n, p, N, directed=False):
    """Creates N PossionNetworks of n nodes with probability p of link formation.
    PossionNetworks are undirected by default but can be made directed. Then calculates
    the average node degree distribution and the distribution of total link counts
    over the population of N networks and compares this to those predicted by
    combinatorics arguments. Creates a figure graphing the comparison between these
    distributions. Returns a pair of times representing the start and end of the calculations."""
    start = time()
    networks = [PoissonNetwork(n, p, directed=directed) for _ in range(N)]

    #calculate average and predicted total link distributions
    pred_total_dist, real_total_dist = networks[0].pred_total_link_dist(), []
    for m in range(n*(n-1)//2*(2 if directed else 1)):
        real_total_dist.append(sum([network.total_links()==m for network in networks])/N)

    #calculate average and predicted degree distributions
    pred_link_dist, real_link_dist = networks[0].pred_degree_dist(), []
    for d in range(n):
        real_link_dist.append(sum([network.degree_dist()[d] for network in networks])/N)
    if directed:
        real_inlink_dist = []
        for d in range(n):
            real_inlink_dist.append(sum([network.degree_dist(False)[d] for network in networks])/N)
    end = time()

    #create figure
    plt.subplot(3 if directed else 2,1,1)
    plt.plot(range(n*(n-1)//2*(2 if directed else 1)),real_total_dist, label="Real")
    plt.plot(range(n*(n-1)//2*(2 if directed else 1)),pred_total_dist, label="Predicted")
    plt.title("Distribution of Total Links")
    plt.ylabel("Frequency")
    plt.xlabel("Total Links")
    plt.legend()
    plt.subplot(3 if directed else 2,1,2)
    plt.plot(range(n),real_link_dist, label="Real")
    plt.plot(range(n),pred_link_dist, label="Predicted")
    plt.title(("Out-" if directed else "") + "Degree Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Degree")
    plt.legend()
    if directed:
        plt.subplot(3 if directed else 2,1,3)
        plt.plot(range(n),real_inlink_dist, label="Real")
        plt.plot(range(n),pred_link_dist, label="Predicted")
        plt.title("In-Degree Distribution")
        plt.ylabel("Frequency")
        plt.xlabel("Degree")
        plt.legend()
    plt.tight_layout()
    plt.show()
    return start, end



medici()
