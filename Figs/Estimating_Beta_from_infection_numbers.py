from math import exp, factorial

import EoN
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar

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

#estimates the size of I[t+tau] using method 1
def method_1_applied(I0, S0, N, ave_k, lambd, beta, tau):

    Is = [I0]
    Ss = [S0]
    Its = [round(I0*(1-rn(0, tau, lambd, beta)))]
    for step in range(1, 6):
        Is.append(method_1(Is[-1], Ss[-1], N, ave_k, pn(step-1, tau, lambd, beta)))
        Ss.append(Ss[-1]-Is[-1])
        Its.append(round(Is[-1]*(1-rn(step, tau, lambd, beta))))

    return sum(Its)

#estimates the size of I[t+tau] using method 2
def method_2_applied(I0, S0, N, ave_k, lambd, beta, tau):

    Is = [I0]
    Ss = [S0]
    Its = [round(I0 * (1 - rn(0, tau, lambd, beta)))]
    for step in range(1, 6):
        Is.append(method_2(Ss[-1], ave_k, N, Is[-1], pn(step - 1, tau, lambd, beta)))
        Ss.append(Ss[-1] - Is[-1])
        Its.append(round(Is[-1] * (1 - rn(step, tau, lambd, beta))))

    return sum(Its)

def error_in_prediction_method_1(beta, num_infectious, num_susceptible, tau, N, ave_k, lambd):

    if len(num_infectious) <= 1:
        raise ValueError('num_infectious is too small in length')
    if len(num_susceptible) != len(num_infectious):
        raise ValueError('num_susceptible and num_infectious are not equal in length')

    predicted_values = [method_1_applied(num_infectious[i], num_susceptible[i], N, ave_k, lambd, beta, tau) for i in range(len(num_infectious)-1)]

    error = 0
    for i in range(len(predicted_values)):
        error += (predicted_values[i]/num_infectious[i+1] - 1)**2

    return error**(1/2)


def error_in_prediction_method_2(beta, num_infectious, num_susceptible, tau, N, ave_k, lambd):

    if len(num_infectious) <= 1:
        raise ValueError('num_infectious is too small in length')
    if len(num_susceptible) != len(num_infectious):
        raise ValueError('num_susceptible and num_infectious are not equal in length')

    predicted_values = [method_2_applied(num_infectious[i], num_susceptible[i], N, ave_k, lambd, beta, tau) for i in range(len(num_infectious)-1)]

    error = 0
    for i in range(len(predicted_values)):
        error += (predicted_values[i]/num_infectious[i+1] - 1)**2

    return error**(1/2)


def beta_predictor_infection_numbers(sim, tau=0.05, lambd=1, threshold_for_infection=0.8):

    N = sim.G.number_of_nodes()
    ave_k = 2*sim.G.number_of_edges()/N
    t, D = sim.summary()
    S, I, R = D['S'], D['I'], D['R']

    if S[-1] >= threshold_for_infection*N:
        raise ValueError('Epidemic does not occur')

    max_time_index = np.where(np.isclose(S, round(N*threshold_for_infection)))
    max_time = t[max_time_index][0]
    max_time_sample = max_time - max_time % tau
    times = np.linspace(0, max_time_sample, int(max_time_sample/tau + 1))
    num_susceptible, num_infectious = EoN.subsample(times, t, S, I)

    beta_hat_1 = minimize_scalar(lambda x: error_in_prediction_method_1(x, num_infectious, num_susceptible, tau, N, ave_k, lambd), bounds=(0, 100)).x
    beta_hat_2 = minimize_scalar(lambda x: error_in_prediction_method_2(x, num_infectious, num_susceptible, tau, N, ave_k, lambd), bounds=(0, 100)).x
    return(beta_hat_1, beta_hat_2)



