from operator import itemgetter
from math import exp, floor
from scipy.optimize import root_scalar
# aim is to take in a graph - run a bunch of simulations on that graph? and collate the results for analysis somehow?

#inv_cfs = inv_cf(plaw_mean_max(6, 100, range_lower=1))
#graph = assortative_graph(N=10000, num_types=4, node_type_matrix=[[0.8, 0.1, 0.05, 0.05],[0.1, 0.8, 0.05, 0.05],[0.05, 0.05, 0.9, 0],[0.05, 0.05, 0, 0.9]])
# graph = networkx.erdos_renyi_graph(10000, 6/10001)
#
# tau = 0.5
# gamma = 1

# t, S, I, R = EoN.fast_SIR(graph, tau=tau, gamma=gamma, return_full_data=False)

# given sim object we can get graph and  transmissions history


def actual_R_by_time(sim):

    '''Given a simulation object from EoN will return times, num_infect, prop_infect where times is a list of recovery times for node,
    num_infect is the number of nodes that were infected by a node, and prop_infect is the proportion of neighbours that a node infected. Note node ID is not returned.'''

    node_R_dic = {node:[sim.node_history(node)[0][-1],0, sim.G.degree(node)] for node in sim.G.nodes() if sim.node_history(node)[1][-1] == 'R'}
    for (time, source, target) in sim.transmissions():
        if source is not None:
            node_R_dic[source][1] += 1

    times = []
    num_infect = []
    neighbours = []
    for time, infecteds, nbrs in sorted(node_R_dic.values(), key=itemgetter(0)):
            times.append(time)
            num_infect.append(infecteds)
            neighbours.append(nbrs)

    return times, num_infect, neighbours

def time_bucketing(t, S, I, R, t_window_length):
    if t[-1]/t_window_length <= 3:
        raise ValueError("The epidemic is too short to generate buckets")

    num_buckets = floor(max(t)/t_window_length)
    S_bucketed, I_bucketed, R_bucketed = [0]*num_buckets, [0]*num_buckets, [0]*num_buckets

    bucket = 0
    for idx in range(len(t)-1):
        # if time and next time along are inside our bucket it works nicely
        if t[idx+1] < t_window_length*(bucket+1):
            S_bucketed[bucket] += S[idx]*(t[idx+1]-t[idx])
            I_bucketed[bucket] += I[idx]*(t[idx+1]-t[idx])
            R_bucketed[bucket] += R[idx]*(t[idx+1]-t[idx])
        # if time and next time along are across a bucket boundary that needs to be accounted for
        else:
            S_bucketed[bucket] += S[idx]*(t_window_length*(bucket+1)-t[idx])
            I_bucketed[bucket] += I[idx]*(t_window_length*(bucket+1)-t[idx])
            R_bucketed[bucket] += R[idx]*(t_window_length*(bucket+1)-t[idx])

            S_bucketed[bucket+1] += S[idx]*(t[idx+1]-t_window_length*(bucket+1))
            I_bucketed[bucket+1] += I[idx]*(t[idx+1]-t_window_length*(bucket+1))
            R_bucketed[bucket+1] += R[idx]*(t[idx+1]-t_window_length*(bucket+1))

            bucket += 1

    # kth bucket covers from k*time_window_length to (k+1)*time_window_length. So time could start at either end of the bucket or the midpoint. Chose midpoint
    new_t = [t_window_length*(k+1)/2 for k in range(num_buckets)]

    return new_t, S_bucketed, I_bucketed, R_bucketed

def beta_solver(I_t0, I_t1, lambd, ave_deg, tau):

    def eq_to_solve(beta):
        return beta / (beta + lambd) * (1 - exp(-(beta+lambd)*tau)) - (I_t1/I_t0 - exp(-tau))/(ave_deg-1)

    return root_scalar(eq_to_solve, x0=0.5, method='secant').root

def Rt_estimator(I, lambd, ave_deg, tau):
    Rt_estimates = []
    for idx in range(len(I)-1):
        if I[idx] == 0 or I[idx+1] == 0:
            Rt_estimates.append(0)
        else:
            beta = beta_solver(I[idx], I[idx+1], lambd, ave_deg, tau)
            Rt_estimates.append(beta/(beta + lambd)*(ave_deg-1))
    Rt_estimates.append(0)
    return Rt_estimates

#
# temp_t, temp_S, temp_I, temp_R = time_bucketing(t, S, I, R, 0.1)
# plt.scatter(temp_t, Rt_estimator(temp_I, 1, 5, 0.3))
# plt.show()
# plt.plot(t, I, color='red')
# plt.plot(t, S, color='green')
# plt.plot(t, R, color='black')
# plt.show()
#
