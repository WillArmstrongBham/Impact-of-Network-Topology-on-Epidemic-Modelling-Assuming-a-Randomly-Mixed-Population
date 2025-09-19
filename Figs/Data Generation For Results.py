import EoN
import networkx as nx
from random import random
from Estimating_Beta_from_infection_numbers import beta_predictor_infection_numbers
import graph_generation_methods as ggm
import numpy as np
import pandas as pd
import time

def forecasting_final_size(S, I, R, time_index, ave_k, beta, lambd):

    St = S[time_index]
    It = I[time_index]
    Rt = R[time_index]

    SIt = It * (ave_k - 1) * St / (St + It +Rt)
    SSt = St * ave_k - SIt - Rt * (ave_k - 1) * beta / (beta + lambd)

    tf, Sf, If, Rf = EoN.SIR_homogeneous_pairwise(St, It, Rt, SIt, SSt, ave_k, beta, lambd)

    return Rf[-1]


lambd = 1
threshold_for_infection = 0.9
tau = 0.05

df = []

start = time.time()

for a in range(30):
    N = 10000
    ave_k = 5

    G = ggm.scale_free_graph(N, ave_k)

    G = nx.subgraph(G, max(nx.connected_components(G), key=len)).copy()

    N = G.number_of_nodes()
    ave_k = 2*G.number_of_edges()/N

    for b in range(20):

        beta = random()*10

        sim = EoN.fast_SIR(G, beta, lambd, rho=10/N, return_full_data=True)
        (t, D) = sim.summary()
        S, I, R = D['S'], D['I'], D['R']

        if S[-1] >= threshold_for_infection*N:
            continue

        #max time is the limit for which we gather data from most parameter estimation methods
        max_time_index = np.where(np.isclose(S, round(N*threshold_for_infection)))[0][0]
        max_time = t[max_time_index]
        if max_time < 2*tau:
            continue

        #method for estimation from node infections
        total_number_infected = 0
        total_number_infectable = 0
        nodes_infected_by_nodes = {node: [] for node in G.nodes()}
        for t, u, v in sim.transmissions():
            if u is not None:
                nodes_infected_by_nodes[u].append(v)
        for node in G.nodes():
            node_history = sim.node_history(node)
            if len(node_history[1]) < 3:
                continue

            nbrs = list(G[node].keys())

            t_recovered = node_history[0][2]
            number_still_susceptible_nbrs = 0
            for nbr, status in sim.get_statuses(nbrs, t_recovered).items():
                if status == 'S':
                    number_still_susceptible_nbrs += 1

            total_number_infected += len(nodes_infected_by_nodes[node])
            total_number_infectable += len(nodes_infected_by_nodes[node]) + number_still_susceptible_nbrs
        alpha_hat = total_number_infected / total_number_infectable
        beta_hat_node_infections = alpha_hat*lambd/(1-alpha_hat)


        #using the method from infection numbers, ths is likely to be very slow
        beta_hat_Inums_sus_nbrs, beta_hat_Inums_inf_nbrs = beta_predictor_infection_numbers(sim, tau=tau, lambd=lambd, threshold_for_infection=threshold_for_infection)

        actual_final_size = R[-1]
        beta_fcast = forecasting_final_size(S, I, R, max_time_index, ave_k, beta, lambd)

        beta_hat_node_infections_fcast = forecasting_final_size(S, I, R, max_time_index, ave_k, beta_hat_node_infections, lambd)

        beta_hat_Inums_sus_nbrs_fcast = forecasting_final_size(S, I, R, max_time_index, ave_k, beta_hat_Inums_sus_nbrs, lambd)

        beta_hat_Inums_inf_nbrs_fcast = forecasting_final_size(S, I, R, max_time_index, ave_k, beta_hat_Inums_inf_nbrs, lambd)

        actual_final_size = R[-1]

        df.append([beta, beta_fcast, actual_final_size, beta_hat_node_infections, beta_hat_node_infections_fcast, beta_hat_Inums_sus_nbrs, beta_hat_Inums_sus_nbrs_fcast, beta_hat_Inums_inf_nbrs, beta_hat_Inums_inf_nbrs_fcast])


df = pd.DataFrame(df)
df.columns = ['Actual Beta', 'Actual Beta Forecast', 'Actual Final Size', 'Beta Hat Node Infections', 'Beta Hat Node Infections Forecast', 'Beta Hat Infection num sus nbrs', 'Beta Hat Infection num sus nbrs Forecast', 'Beta Hat Infection num inf nbrs', 'Beta Hat Infection num inf nbrs Forecast' ]

stop = time.time()

print(stop-start)

df.to_csv("SF configuration Graph data.csv", index=False, encoding='utf-8')
