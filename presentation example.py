import random

import EoN
from matplotlib import pyplot as plt

from SIR_simulation_for_graph import actual_R_by_time
import networkx as nx
import numpy as np
from statistics import median, mean

gamma = 1
epidemic_threshold = 0.75

actual_taus = []
lower_taus = []
upper_taus = []

for a in range(10):
    N= 10000
    ave_k = 6
    G = nx.erdos_renyi_graph(N, ave_k/N)
    # G = nx.barabasi_albert_graph(N, ave_k)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy() #making graph be largest connected component - this does change N and ave-k how to change?
    N = G.number_of_nodes()
    ave_k = 2*G.number_of_edges()/N
    prop_infected = 10 / N

    for b in range(100):


        tau = random.random()*10 #chance of infection between two edges
        sim = EoN.fast_SIR(G, tau=tau, rho=prop_infected, gamma=gamma, return_full_data=True)
        (t, D) = sim.summary(nodelist=G.nodes())
        S, I, R = D['S'], D['I'], D['R']
        #want to check if an epidemic has occured
        if S[-1] >= epidemic_threshold*N:
            continue

        #lists for calculation of tau

        tau_upper_susceptible = 0
        tau_infected = 0
        tau_lower_susceptible = 0

        # generates a dictionary of {node: [list of nodes that the node infected]}
        nodes_infected_by_nodes = {node:[] for node in G.nodes()}
        for t, u, v in sim.transmissions():
            if u is not None:
                nodes_infected_by_nodes[u].append(v)
        # value of tau for each node
        for node in G.nodes():
            node_history = sim.node_history(node)
            # check node was infected
            if len(node_history[1]) < 3:
                continue

            nbrs = list(G[node].keys())

            #get nbrs that are susceptible at start of time

            t_infected = node_history[0][1]
            nbrs_initially_susceptible = []
            for nbr, status in sim.get_statuses(nbrs, t_infected).items():
                if status == 'S':
                    nbrs_initially_susceptible.append(nbr)

            #gets nbrs that are susceptible when node recovers
            t_recovered = node_history[0][2]
            nbrs_recovery_susceptible = []
            for nbr, status in sim.get_statuses(nbrs, t_recovered).items():
                if status == 'S':
                    nbrs_recovery_susceptible.append(nbr)

            #add the total number of infected
            tau_infected += len(nodes_infected_by_nodes[node])
            #add largest amount of susceptible for lower bound of tau
            tau_lower_susceptible += len(nbrs_initially_susceptible)
            #add smallest amount of susceptibles/actually infected for upper bound of tau
            tau_upper_susceptible += len(nbrs_recovery_susceptible)+len(nodes_infected_by_nodes[node])

        #alpha is probability of infection along edge
        alpha_lower = tau_infected/tau_lower_susceptible
        alpha_upper = tau_infected/tau_upper_susceptible

        actual_taus.append(tau)
        lower_taus.append(alpha_lower*gamma/(1-alpha_lower))
        upper_taus.append(alpha_upper*gamma/(1-alpha_upper))

        #goal is to predict tau
        #if a vertex gets infected then look at how many susceptible neighbours it has at time of infection
        # look at how many it has infected at time of recovery
        # look at how many susceptible neighbours it has at time of recovery

plt.rcParams.update({'text.usetex': True,
                     'font.family': 'serif',
                     'font.serif': 'Computer Modern',
                     'font.size': 14,
                     'figure.figsize':(6,6),
                    'axes.labelsize': 16,
                    'axes.titlesize':16})


plt.scatter(actual_taus, upper_taus, c='r', label='Upper Estimate', alpha=0.5, s=2)
plt.scatter(actual_taus, lower_taus, c='b', label='Lower Estimate', alpha=0.5, s=2)
plt.legend()
plt.xlabel(r"Actual value of $\beta$")
plt.ylabel(r"Estimated value of $\beta$")
plt.title(r"Prediction of $\beta$ for an Erdos-Renyi Graph")
plt.savefig("Node Infections Prediction for an Erdos-Renyi Graph.pdf", bbox_inches='tight')



