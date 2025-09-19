from math import exp, factorial
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

ave_k = 6
beta = 1
lambd = 1

N = 10000
I0_prop = 0.1
I0 = N*I0_prop
S_prop = 0.85
S = N*S_prop


def p1(tau):
    return beta/(beta+lambd) * (1-exp(-1*(beta+lambd)*tau))
def h_of_k(h, k):
    prob = 1
    for i in range(0, h):
        prob *= (I0 - i)
    for i in range(0, k-h):
        prob *= (N-1-I0-i)
    for i in range(0, k):
        prob /= (N-1-i)
    prob *= factorial(k)
    prob /= factorial(h)
    prob /= factorial(k-h)

    return prob
def exp_pv(tau, k=ave_k):
    prob = 0
    for h in range(0, k+1):
        prob += (1-(1-p1(tau))**h)*h_of_k(h, k)
    return prob
def dist_exp_pv(tau):
    prob = 0
    for k in range(0, int(stats.binom.ppf(0.99, (N-1), ave_k/(N-1)))+1):
        temp_prob = 0
        for h in range(0, k+1):
            temp_prob += (1-(1-p1(tau))**h)*h_of_k(h, k)
        prob += temp_prob*stats.binom.pmf(k, N, ave_k/(N-1))
    return prob


times = np.linspace(0, 3, 300)
method_1 = [p1(x)*I0*(ave_k-1) for x in times]
method_2 = [exp_pv(x)*S for x in times]
method_3 = [dist_exp_pv(x)*S for x in times]

plt.plot(times, method_1, color='red')
plt.plot(times, method_2, color='blue')
plt.plot(times, method_3, color='green')
plt.show()
