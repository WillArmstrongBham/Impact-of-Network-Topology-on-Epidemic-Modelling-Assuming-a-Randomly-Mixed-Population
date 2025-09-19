from math import exp, factorial
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns

#example values
N = 10000
I = 1000
S = 8500
ave_k = 5

lambd = 1
beta = 1

#need to generate In

#generates probability of v_n infecting v_n-1
def pn(n, tau, lambd, beta):
    if tau == 0:
        return 0
    if n == 0:
        return beta/(beta+lambd) * (1-exp(-(beta+lambd)*tau))
    else:
        denom = 0
        for i in range(n):
            denom += (beta+lambd)**i * tau**i / factorial(i)
        num = denom + ((beta+lambd)**n * tau**n / factorial(n))

        return beta/(beta+lambd) * (1-exp(-(beta+lambd)*tau)*num) / (1 - exp(-(beta+lambd)*tau)*denom)
#generates probability of v_n recovering
def rn(n, tau, lambd, beta):
    if tau == 0:
        return 0
    if n ==0:
        return 1 - exp(-lambd*tau)
    else:
        denom = 0
        nom = 0
        for i in range(n):
            nom += tau**i * beta**i / factorial(i)
            denom += (beta+lambd)**i * tau**i / factorial(i)

        return 1 - ((beta+lambd)**n / (beta**n) * exp(-lambd*tau) * (1-exp(-beta*tau)*nom) / (1 - exp(-(beta+lambd)*tau)*denom))
#generates size of In+1 from In using method 1
def method_1(I, S, N, ave_k, p):
    return round((ave_k-1)*p*I*(S/N))
#method two uses two prior functions to make life easier - conditioning stuff from diss
def prob_i_given_num_nbrs(num_nbrs, i, N, I):
    prob = 1
    if num_nbrs == 0:
        return 0
    if i > num_nbrs:
        return 0
    for j in range(num_nbrs):
        if j <= i-1:
            prob *= (I-j)
        if j <= num_nbrs-i-1:
            prob *= (N-1-I-j)
        prob /= (N-1-j)
    prob *= factorial(num_nbrs)/(factorial(i)*factorial(num_nbrs-i))

    return prob

def inf_nbr_pmf(ave_k, N, I):

    max_nbrs = int(stats.binom.ppf(0.995, N, ave_k/N)) #actually max nbrs -1 since this is number of infected nbrs
    infected_pmf = [0]*max_nbrs
    for i in range(max_nbrs):
        prob_i_nbrs = 0
        for num_nbrs in range(max_nbrs):
            prob_i_nbrs += stats.binom.pmf(num_nbrs, N, ave_k/N) * prob_i_given_num_nbrs(num_nbrs, i, N, I)
        infected_pmf[i] = prob_i_nbrs

    return infected_pmf

def method_2(S, ave_k, N, I, pn):

    infected_pmf = inf_nbr_pmf(ave_k, N, I)
    prob_s_in_I = 0
    for num_i in range(len(infected_pmf)):
        prob_s_in_I += infected_pmf[num_i] * (1 - (1 - pn)**num_i)

    return round(prob_s_in_I*S)


def method_1_applied(I0, S0, t_max):

    I_by_tau = []
    times = np.linspace(0, t_max, round(50*t_max))

    for tau in times:

        Is = [I0]
        Ss = [S0]
        Its = [round(I0*(1-rn(0, tau, lambd, beta)))]
        for step in range(1, 6):
            Is.append(method_1(Is[-1], Ss[-1], N, ave_k, pn(step-1, tau, lambd, beta)))
            Ss.append(Ss[-1]-Is[-1])
            Its.append(round(Is[-1]*(1-rn(step, tau, lambd, beta))))
        I_by_tau.append(Its)

    return times, I_by_tau

def method_2_applied(I0, S0, t_max):

    I_by_tau = []
    times = np.linspace(0, t_max, round(50*t_max))
    for tau in times:

        Is = [I0]
        Ss = [S0]
        Its = [round(I0 * (1 - rn(0, tau, lambd, beta)))]
        for step in range(1, 6):
            Is.append(method_2(Ss[-1], ave_k, N, Is[-1], pn(step - 1, tau, lambd, beta)))
            Ss.append(Ss[-1] - Is[-1])
            Its.append(round(Is[-1] * (1 - rn(step, tau, lambd, beta))))
        I_by_tau.append(Its)

    return  times, I_by_tau

times, df = method_2_applied(I, S, 5)

df = pd.DataFrame(df)
df['Time'] = times


plt.stackplot(df['Time'], df[0], df[1], df[2], df[3], df[4], df[5])
plt.show()