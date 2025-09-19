# aim of this file is to have a single place for all of the properties of a graph I am interested in
# those include: distribution of degrees, connected components, clustering, small world porperty, assortativity, centrality, node reslience perhaps?

import networkx as nx
import matplotlib.pyplot as plt

graph = nx.lollipop_graph(12, 6)

def degree_distribution_hist(graph):
    degrees = nx.degree_histogram(graph)
    degree_height = [x / sum(degrees) for x in degrees]
    positions = range(len(degrees))
    plt.bar(positions, degree_height ,width=1, align='edge')
    plt.xlabel('Degree')
    plt.show()

def connected_components_distribtion(graph):
    len_components = [len(c) for c in nx.connected_components(graph)]
    plt.hist(len_components, range=(0, max(len_components)+1))
    plt.show()

def connected_component_largest(graph):
    size_of_largest = len(max(nx.connected_components(graph), key=len))
    perc_nodes = size_of_largest / graph.number_of_nodes()
    return (size_of_largest, f'{perc_nodes*100:.2f}% of nodes in largest connected component')

# clustering in terms of triangles/triads use
# nx.transitivity(graph)

def clustering_distribution(graph):
    clustering_values = nx.clustering(graph).values()
    plt.hist(clustering_values, range=(0, 1), density=True, bins=len(clustering_values))
    plt.show()

# for average of clustering coeffificients use
# nx.average_clustering

def average_distance(graph):
    total_total = 0
    total_count = 0
    for a in nx.all_pairs_shortest_path_length(graph):
        total_total += sum(a[1].values())
        total_count += len(a[1].values())-1

    return total_total/total_count

def average_distance_distribution(graph):
    ave_dist_list = []
    for a in nx.all_pairs_shortest_path_length(graph):
        total = sum(a[1].values())
        count = len(a[1].values()) - 1
        ave_dist_list.append(total/count)

    plt.hist(ave_dist_list, range=(0, max(ave_dist_list)+1), density=True)
    plt.show()

# for assortativity by node degree use the following
# nx.degree_assortativity_coefficient(graph)



