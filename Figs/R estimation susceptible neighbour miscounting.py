import EoN
import networkx as nx
import random
from collections import Counter
import pandas as pd

gamma = 1
epidemic_threshold = 0.75
sample_numbers = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]

dfN = []
dfave_k = []
dfsample = []
dfsus_nbrs = []
dfS = []
dfnum1 = []
dfnum2 = []
dfnum3 = []
dfnum4 = []
dfnum5_or_more = []


Ns = [10000, 50000]

for a in range(6):
    N = Ns[a % len(Ns)]
    ave_k = 5

    G = nx.erdos_renyi_graph(N, ave_k/N)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    N = G.number_of_nodes()
    ave_k = 2*G.number_of_edges()/N
    prop_infected = 10/N

    for b in range(100):

        tau = random.random()*2+0.5
        sim = EoN.fast_SIR(G, tau=tau, rho=prop_infected, gamma=gamma, return_full_data=True)
        (t, D) = sim.summary(nodelist=G.nodes())
        S, I, R = D['S'], D['I'], D['R']

        if S[-1] >= epidemic_threshold*N:
            continue

        for sample in sample_numbers:
            if max(I) >= round(sample*N):
                time_index = list(I).index(int(round(sample*N)))
                time = t[time_index]
                infected_nodes = []
                adjacent_susceptible_nodes = []
                for key, value in sim.get_statuses(G.nodes(),time).items():
                    if value =='I':
                        infected_nodes.append(key)

                for node in infected_nodes:
                    for nbr in G.neighbors(node):
                        if sim.node_status(nbr, time) == 'S':
                            adjacent_susceptible_nodes.append(nbr)

                nbr_double_cnt = [0, 0, 0, 0, 0]
                for key, value in Counter(Counter(adjacent_susceptible_nodes).values()).items():
                    if key <= 4:
                        nbr_double_cnt[key-1] += value
                    else:
                        nbr_double_cnt[4] += value

                dfN.append(N)
                dfave_k.append(ave_k)
                dfsample.append(sample)
                dfsus_nbrs.append(len(set(adjacent_susceptible_nodes)))
                dfS.append(S[time_index])
                dfnum1.append(nbr_double_cnt[0])
                dfnum2.append(nbr_double_cnt[1])
                dfnum3.append(nbr_double_cnt[2])
                dfnum4.append(nbr_double_cnt[3])
                dfnum5_or_more.append(nbr_double_cnt[4])


df = pd.DataFrame([dfN, dfave_k, dfsample, dfsus_nbrs, dfS, dfnum1, dfnum2, dfnum3, dfnum4, dfnum5_or_more])
df = df.transpose()
df.columns= ['N', 'ave_k', 'sample', 'susceptible neighbours', 'S at time of sample', 'num 1s', 'num 2s', 'num 3s', 'num 4s', 'num 5 or more']
df.to_csv(r'R estimation susceptible neighbour miscounting data.csv', encoding='utf-8', index=False)




