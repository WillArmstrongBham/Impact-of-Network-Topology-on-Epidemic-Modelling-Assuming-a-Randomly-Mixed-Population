import EoN
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import random
import pandas as pd
import seaborn as sns

lambd = 1
epidemic_threshold = 0.8

plt.rcParams.update({'text.usetex': True,
                     'font.family': 'serif',
                     'font.serif': 'Computer Modern',
                     'font.size': 14,
                     'figure.figsize':(6,6),
                    'axes.labelsize': 16,
                    'axes.titlesize':16})

df = []

for b in range(10):
    N = 10000
    ave_k = 5

    G = nx.fast_gnp_random_graph(N, ave_k/N)
    G = nx.subgraph(G, max(nx.connected_components(G), key=len)).copy()

    N = G.number_of_nodes()
    ave_k = 2*G.number_of_edges()/N

    for a in range(10):

        beta = random()*8+0.25

        t, S, I, R = EoN.fast_SIR(G, beta, 1, rho=10/N)

        if S[-1] > epidemic_threshold*N:
            continue

        time_index = np.where(np.isclose(S, round(N*epidemic_threshold)))[0][0]

        St = S[time_index]
        It = I[time_index]
        Rt = R[time_index]

        SIt = It*(ave_k-1)*St/N
        SSt = St*ave_k - SIt - Rt*(ave_k-1)*beta/(beta+lambd)

        tf, Sf, If, Rf = EoN.SIR_homogeneous_pairwise(St, It, Rt, SIt, SSt, ave_k, beta, lambd)

        new_times = t[time_index:]
        new_times = new_times - new_times[0]

        df.append([beta, R[-1]-Rf[-1]])

        plt.plot(tf, EoN.subsample(tf, new_times, R[time_index:])-Rf, color='grey', alpha=0.2)

plt.xlim(0, 15)
plt.xlabel('Time Since Epidemic Threshold')
plt.ylabel(r"Actual $R[t]$ - Predicted $R[t]$")
plt.title("Accuracy of Forecasting Recovered Time Series")
plt.tight_layout()
plt.savefig('Forecasting_Algorithm_Time_Series.pdf', bbox_inches='tight')
plt.close()

df = pd.DataFrame(df)
df.columns = [r'Infection Rate $\beta$', 'Final $R[t]$ - Final Predicted $R[t]$']
sns.scatterplot(df, x=r'Infection Rate $\beta$', y='Final $R[t]$ - Final Predicted $R[t]$')
plt.title(r'Accuracy of Forecasting Depending on $\beta$')
plt.tight_layout()
plt.savefig('Forecasting_Algorithm_Beta_Dependent.pdf', bbox_inches='tight')