import EoN
import networkx as nx
from matplotlib import rc
import matplotlib.pyplot as plt

import random
import numpy as np

colors = ['#5AB3E6','#FF2000','#009A80','#E69A00', '#CD9AB3', '#0073B3','#F0E442']

rc('text', usetex=True)

rho = 0.0250
target_k = 6
N = 10000
tau = 0.5
gamma = 1.
ts = np.arange(0, 40, 0.05)
count = 50

def generate_network(Pk, N, ntries=100):
    r'''Generates an N-node random network whose degree distribution is given by Pk'''
    counter = 0
    while counter < ntries:
        counter += 1
        ks = []
        for ctr in range(N):
            ks.append(Pk())
        if sum(ks)%2==0:
            break
    if sum(ks)%2==1:
        raise EoN.EoNError("cannot generate even degree sum")
    G = nx.configuration_model(ks)
    return G

# Poisson distribution and generating function(?) used in ER graph

def PKPoisson():
    return np.random.poisson(target_k)
def PsiPoisson(x):
    return np.exp(-target_k*(1-x))
def DPsiPoisson(x):
    return target_k*np.exp(-target_k*(1-x))

# Distribtuion for regular network

def PkHomogenous():
    return target_k
def PsiHomogenous(x):
    return x**target_k
def DPsiHomogenous(x):
    return target_k*x**(target_k-1)

# Distribution for truncated power network note that this is dependent on target k - the values of things will have to change if that changes

assert(target_k==6)

PlPk = {}
exponent = 1.5
kave = 0
for k in range(1, 61):
    PlPk[k] = k**(-exponent)
    kave += k*PlPk[k]

normfactor = sum(PlPk.values())
for k in PlPk:
    PlPk[k] /= normfactor

def PkPowLaw():
    r = random.random()
    for k in PlPk:
        r -= PlPk[k]
        if r <0:
            return k

def PsiPowLaw(x):
    # print PlPk
    rval = 0
    for k in PlPk:
        rval += PlPk[k]*x**k
    return rval

def DPsiPowLaw(x):
    rval = 0
    for k in PlPk:
        rval += k*PlPk[k]*x**(k-1)
    return rval

# end of power law distributions

def process_degree_distribution(N, Pk, color, Psi, DPsi, symbol, label, count):
    report_times = np.linspace(0, 30, 3000)
    sums = 0*report_times
    for cnt in range(count):
        G = generate_network(Pk, N)
        t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho)
        plt.plot(t, I*1./N, '-', color=color, alpha=0.1, linewidth=1)
        subsampled_I = EoN.subsample(report_times, t, I)
        sums += subsampled_I*1./N
    ave=sums/count
    plt.plot(report_times, ave, color='k')

    # Do EBCM
    N=G.order()
    t, S, I, R = EoN.EBCM_uniform_introduction(N, Psi, DPsi, tau, gamma, rho, tmin=0, tmax=10, tcount=41)
    plt.plot(t, I/N, symbol, color=color, markeredgecolor='k', label=label)

    for cnt in range(3):
        G = generate_network(Pk, N)
        t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho)
        plt.plot(t, I*1./N, '-', color='k', linewidth=1)



plt.figure(figsize=(8,4))

# Power Law
process_degree_distribution(N, PkPowLaw, colors[3], PsiPowLaw, DPsiPowLaw, 'd', r'Truncated Power Law', count)
# Poisson
process_degree_distribution(N, PKPoisson, colors[0], PsiPoisson, DPsiPoisson, '^', r'Erd\H{o}s--R\'{e}nyi', count)
# Homogenous
process_degree_distribution(N, PkHomogenous, colors[2], PsiHomogenous, DPsiHomogenous, 's', r'Homogenous', count)

plt.xlabel(r'$t$', fontsize=12)
plt.ylabel(r'Proportion infected', fontsize=12)
plt.legend(loc='upper right', numpoints=1)

plt.axis(xmax=10, xmin=0, ymin=0)
plt.show()