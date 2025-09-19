import networkx as nx
import EoN
import pandas as pd
from random import random
import Estimating_Beta_from_infection_numbers as EBFIN

N = 10000
ave_k = 5
lambd = 1
df = []

for a in range(100):

    N = 10000
    ave_k = 5

    G = nx.erdos_renyi_graph(N, ave_k/N)
    G = nx.subgraph(G, max(nx.connected_components(G), key=len)).copy()

    N = G.number_of_nodes()
    ave_k = G.number_of_edges()*2/N

    for b in range(1):

        tau = random()*4.5 + 0.25
        sim = EoN.fast_SIR(G, tau, gamma = lambd, rho = 10/N, return_full_data=True)

        if sim.summary()[1]['S'][-1] >= sim.G.number_of_nodes()*0.8:
            continue

        betas = EBFIN.beta_predictor_infection_numbers(sim)
        df.append([tau, betas[0], betas[1]])

df = pd.DataFrame(df, columns=['Actual eta', 'Method 1 Beta', 'Method 2 Beta'])

df.to_csv('Estimating Beta from infection numbers.csv', index=False, encoding='utf-8')
