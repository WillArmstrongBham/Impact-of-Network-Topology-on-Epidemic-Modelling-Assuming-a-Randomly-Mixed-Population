from truncated_power_distribution import plaw_mean_max, inv_cf, sample_inv_cf
import numpy as np
import random
import networkx as nx


def l2_square(a, b):
    '''Temp dist function set to be euclidian distance squared'''
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


inv_cfs = inv_cf(plaw_mean_max(6, 100))


def closest_neighbours(inv_cfs, size, picking_radius, dist=l2_square):
    '''Generates a square lattice in 2d with side length of size where each point on the lattice is given a maximum degree samples from inv_cfs an inverse cumulative function dictionary.
    Points are then picked randomly on the grid and edges are formed to the closest neighbours, provided that the neighbours have free edges. The distance limit for
    neighbours is given by picking_radius and dist function. A networkx graph is returned. Note the graph is unlikely to be connected. '''

    # coords are written (i, j) and numpy array is accessed by array[j][i] unsure if this is correct to do
    spatial_grid = np.array([{'max_edges': sample_inv_cf(inv_cfs), 'cur_edges': 0, 'connected_edges': []} for a in
                             range(size ** 2)]).reshape(size, size)

    unchosen_points = ([(i, j) for i in range(size) for j in range(size)])

    while len(unchosen_points) > 0:

        cur_i, cur_j = unchosen_points.pop(random.randint(0, len(unchosen_points) - 1))

        if spatial_grid[cur_j][cur_i]['cur_edges'] < spatial_grid[cur_j][cur_i]['max_edges']:

            nbrs_at_dist = {}
            for i in range(cur_i - picking_radius, cur_i + picking_radius + 1):
                for j in range(cur_j - picking_radius, cur_j + picking_radius + 1):
                    if 0 <= i < size and 0 <= j < size:
                        if dist((cur_i, cur_j), (i, j)) <= dist((0, 0), (0, picking_radius)):
                            if not (i == cur_i and j == cur_j):
                                if spatial_grid[j][i]['cur_edges'] < spatial_grid[j][i]['max_edges']:
                                    if (i, j) not in spatial_grid[cur_j][cur_i]['connected_edges']:
                                        nbrs_at_dist.setdefault(dist((cur_i, cur_j), (i, j)), []).append((i, j))

            while spatial_grid[cur_j][cur_i]['cur_edges'] < spatial_grid[cur_j][cur_i]['max_edges']:

                if not nbrs_at_dist:
                    break

                min_key = min(nbrs_at_dist)
                min_dis_points = nbrs_at_dist[min_key]
                if len(min_dis_points) == 0:
                    raise KeyError('nbrs_at_dist contains an empty string for a value')

                point = min_dis_points[random.randint(0, len(min_dis_points) - 1)]
                spatial_grid[cur_j][cur_i]['connected_edges'].append(point)
                spatial_grid[cur_j][cur_i]['cur_edges'] += 1
                spatial_grid[point[1]][point[0]]['cur_edges'] += 1
                spatial_grid[point[1]][point[0]]['connected_edges'].append((cur_i, cur_j))

                if len(min_dis_points) == 1:
                    del nbrs_at_dist[min_key]
                else:
                    min_dis_points.remove(point)
                    nbrs_at_dist[min_key] = min_dis_points

    graph_points = []
    graph_edges = []

    for j in range(len(spatial_grid)):
        for i in range(len(spatial_grid[j])):
            graph_points.append((i, j))
            graph_edges.extend([((i, j), p) for p in spatial_grid[j][i]['connected_edges']])

    G = nx.Graph()
    G.add_nodes_from(graph_points)
    G.add_edges_from(graph_edges)
    return G

def closest_neighbors_graph(N, ave_k):
    size = round(N**0.5 + 0.49)

    inv_cfs = inv_cf(plaw_mean_max(ave_k, round((size*ave_k)**0.5)))

    return closest_neighbours (inv_cfs, size, 15)

def distance_weighted_neighbours(inv_cfs, size, picking_radius, k=1 / 2):
    '''Generates a square lattice in 2d with side length of size where each point on the lattice is given a maximum degree samples from inv_cfs an inverse cumulative function dictionary.
    Points are then picked randomly on the grid and edges are formed to random neighbours weighted for distance, provided that the neighbours have free edges. The distance limit for
    neighbours is given by picking_radius l2 distance is assumed. A networkx graph is returned. Note the graph is unlikely to be connected. '''

    # coords are written (i, j) and numpy array is accessed by array[j][i] unsure if this is correct to do
    spatial_grid = np.array([{'max_edges': sample_inv_cf(inv_cfs), 'cur_edges': 0, 'connected_edges': []} for a in
                             range(size ** 2)]).reshape(size, size)

    unchosen_points = ([(i, j) for i in range(size) for j in range(size)])

    while len(unchosen_points) > 0:

        cur_i, cur_j = unchosen_points.pop(random.randint(0, len(unchosen_points) - 1))

        if spatial_grid[cur_j][cur_i]['cur_edges'] < spatial_grid[cur_j][cur_i]['max_edges']:

            nbrs_at_dist = {}
            for i in range(cur_i - picking_radius, cur_i + picking_radius + 1):
                for j in range(cur_j - picking_radius, cur_j + picking_radius + 1):
                    if 0 <= i < size and 0 <= j < size:
                        if l2_square((cur_i, cur_j), (i, j)) <= l2_square((0, 0), (0, picking_radius)):
                            if not (i == cur_i and j == cur_j):
                                if spatial_grid[j][i]['cur_edges'] < spatial_grid[j][i]['max_edges']:
                                    if (i, j) not in spatial_grid[cur_j][cur_i]['connected_edges']:
                                        nbrs_at_dist.setdefault(l2_square((cur_i, cur_j), (i, j)), []).append((i, j))

            while spatial_grid[cur_j][cur_i]['cur_edges'] < spatial_grid[cur_j][cur_i]['max_edges']:

                if not nbrs_at_dist:
                    break

                sorted_distances = sorted(nbrs_at_dist.keys())
                weighted_distances = [d ** (-k) * len(nbrs_at_dist[d]) for d in sorted_distances]
                target_num = random.random() * sum(weighted_distances)
                for i in range(len(weighted_distances)):
                    if target_num < weighted_distances[i]:
                        distance_to_pick = sorted_distances[i]
                    else:
                        target_num -= weighted_distances[i]

                points_to_pick = nbrs_at_dist[distance_to_pick]

                if len(points_to_pick) == 0:
                    raise KeyError('nbrs_at_dist contains an empty string for a value')

                point = points_to_pick[random.randint(0, len(points_to_pick) - 1)]
                spatial_grid[cur_j][cur_i]['connected_edges'].append(point)
                spatial_grid[cur_j][cur_i]['cur_edges'] += 1
                spatial_grid[point[1]][point[0]]['cur_edges'] += 1
                spatial_grid[point[1]][point[0]]['connected_edges'].append((cur_i, cur_j))

                if len(points_to_pick) == 1:
                    del nbrs_at_dist[distance_to_pick]
                else:
                    points_to_pick.remove(point)
                    nbrs_at_dist[distance_to_pick] = points_to_pick

    graph_points = []
    graph_edges = []

    for j in range(len(spatial_grid)):
        for i in range(len(spatial_grid[j])):
            graph_points.append((i, j))
            graph_edges.extend([((i, j), p) for p in spatial_grid[j][i]['connected_edges']])

    G = nx.Graph()
    G.add_nodes_from(graph_points)
    G.add_edges_from(graph_edges)
    return G

def distance_weghted_neighbours_graph(N, ave_k):

    size = round(N**0.5 +0.49)
    inv_cfs = inv_cf(plaw_mean_max(ave_k, round((size*ave_k)**0.5)))

    return distance_weighted_neighbours(inv_cfs, size, 15)


# for sparse random graphs use
# nx.fast_gnp_random_graph()

# for small world ring graphs use either
# nx.newman_watts_strogatz_graph()
# nx.watts_strogatz_graph()

# scale free network using barabasi_albert method
# nx.barabasi_albert_graph()

# method for generating scale free graph with clustering
# nx.powerlaw_cluster_graph()

#configuration model approach - needs even degree and use nx.graph(G), G.remove_edges_from(nx.selfloops(G)) to make desired graph
#nx.configuration_model

def scale_free_graph(N, ave_k):

    inv_cfs = inv_cf(plaw_mean_max(ave_k, round((N*ave_k)**0.5)))
    list_of_degree_edges = [sample_inv_cf(inv_cfs) for a in range(N)]
    if sum(list_of_degree_edges) % 2 == 1:
        list_of_degree_edges[-1] += 1

    G = nx.configuration_model(list_of_degree_edges)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def assortative_graph(N, num_types, node_type_matrix):
    '''
    A method to generate an assortative graph. N is the number of total nodes. num_types is the number of different types of nodes.
    Currently it assumes that the nodes will be evenly distributed and all come from a scale free distribution. node_type_matrix should be a
    square symmetric matrix of size num_types**2 which each row/column summing to 1. Relative proportion of connections between the different types.

    It would be useful to have a list of different sizes for the num_types category. Also an option to choose different distributions for each one.
    '''

    num_node_types = num_types
    # node_type_matrix = [[0.8, 0.05, 0.05, 0.05, 0.05],
    #                     [0.05, 0.8, 0.05, 0.05, 0.05],
    #                     [0.05, 0.05, 0.8, 0.05, 0.05],
    #                     [0.05, 0.05, 0.05, 0.8, 0.05],
    #                     [0.05, 0.05, 0.05, 0.05, 0.8]]

    list_of_nodes = {a: {'connected_edges': [], 'node_type': a % num_node_types, 'maximum_edges': sample_inv_cf(inv_cfs)}
                     for a in range(N)}

    for node in range(N):
        if len(list_of_nodes[node]['connected_edges']) < list_of_nodes[node]['maximum_edges']:
            available_nodes_by_type = {node_type: [] for node_type in range(num_node_types)}
            for other_node in range(N):
                if other_node != node:
                    if len(list_of_nodes[other_node]['connected_edges']) < list_of_nodes[other_node]['maximum_edges']:
                        if other_node not in list_of_nodes[node]['connected_edges']:
                            available_nodes_by_type[list_of_nodes[other_node]['node_type']].append(other_node)

            while len(list_of_nodes[node]['connected_edges']) < list_of_nodes[node]['maximum_edges']:
                weighted_node_types = [
                    len(available_nodes_by_type[node_type]) * node_type_matrix[list_of_nodes[node]['node_type']][node_type]
                    for node_type in range(num_node_types)]
                if sum(weighted_node_types) == 0:
                    break
                cur_sum = random.random() * sum(weighted_node_types)
                for node_type in range(num_node_types):
                    if weighted_node_types[node_type] >= cur_sum:
                        choice = random.choice(available_nodes_by_type[node_type])
                    else:
                        cur_sum -= weighted_node_types[node_type]

                list_of_nodes[choice]['connected_edges'].append(node)
                available_nodes_by_type[list_of_nodes[choice]['node_type']].remove(choice)
                list_of_nodes[node]['connected_edges'].append(choice)

    nodes_for_graph = []
    edges_for_graph = []

    for key, value in list_of_nodes.items():
        nodes_for_graph.append((key, {'node_type': value['node_type']}))
        for edge in value['connected_edges']:
            edges_for_graph.append((key, edge))

    G = nx.Graph()
    G.add_nodes_from(nodes_for_graph)
    G.add_edges_from(edges_for_graph)
    return G

