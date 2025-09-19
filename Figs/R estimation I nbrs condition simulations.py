import EoN
import networkx as nx
import pandas as pd
import random

gamma = 1
epidemic_threshold = 0.75 #when an epidemic occurs
sample_numbers = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.25] #proportion of N infected

#data to be extracted
df = []

for a in range(10):
    N=10000
    ave_k = 5

    G = nx.erdos_renyi_graph(N, ave_k/N)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    N = G.number_of_nodes()
    ave_k = G.number_of_edges()/N
    prop_infected = 10/N

    for b in range(10):
        tau = random.random()*2+0.5
        sim = EoN.fast_SIR(G, tau=tau, rho=prop_infected, gamma=gamma, return_full_data=True)
        (t, D) = sim.summary(nodelist=G.nodes())
        S, I, R = D['S'], D['I'], D['R']

        if S[-1] >= epidemic_threshold*N:
            continue

        for sample in sample_numbers:

            if max(I) >= round(sample*N):
                dist_nbrs = [0] * 20
                time_index = list(I).index(int(round(sample*N)))
                time = t[time_index]
                susceptible_nodes = []

                for key, value in sim.get_statuses(G.nodes(), time).items():
                    if value =='S':
                        susceptible_nodes.append(key)

                for node in susceptible_nodes:
                    infected_nbrs = 0
                    for nbr in G.neighbors(node):
                        if sim.node_status(nbr, time) == 'I':
                            infected_nbrs += 1

                    if infected_nbrs <= 18:
                        dist_nbrs[infected_nbrs] += 1
                    else:
                        dist_nbrs[19] += 1

                df.append([sample, N, round(N*sample), S[time_index]] + dist_nbrs)

df = pd.DataFrame(df)
df.columns = ['Sample', 'N', 'I', 'S', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
df.to_csv(r'R estimation I nbrs condition data.csv', encoding='utf-8', index=False)


