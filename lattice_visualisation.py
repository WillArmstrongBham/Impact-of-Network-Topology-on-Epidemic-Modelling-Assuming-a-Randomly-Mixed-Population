import matplotlib.pyplot as plt
import networkx as nx
import EoN

# example lattices
# G = SF_lattice_closest_nbrs.closest_neighbours(inv_cfs=SF_lattice_closest_nbrs.inv_cfs, size=100, picking_radius=10)
G = SF_lattice_closest_nbrs.distance_weighted_neighbours(inv_cfs=SF_lattice_closest_nbrs.inv_cfs, size=100, picking_radius=10, k=-3)

def lattice_viz(graph):
    positions = {node:node for node in graph.nodes()}
    nx.draw(graph, positions, with_labels=False, node_size=1, node_color='k', alpha=0.1)
    plt.show()

tau = 1
gamma = 1
positions = {node:node for node in G}
sim_kwargs = {'pos':positions}
sim = EoN.fast_SIR(G, tau=tau, gamma=gamma, return_full_data=True, sim_kwargs=sim_kwargs)

ani=sim.animate(ts_plots=['I', 'SIR'], node_size=4)
ani.save('lattice_viz.gif')