import random
from bisect import bisect_right
import warnings
from scipy.optimize import root_scalar

def plaw_exp_max(exponent, range_upper, range_lower=1, output=0):
    """Creates a dictionary of {positive int:probability of positive int} for a discrete truncated power law distribution
    based the exponent and on minimum and maximum values. Output = 0 returns the dictionary. Output = 1 returns
    the average value. Output = 3 returns a tuple of (dict, ave). Use plaw_mean_max to generate a power law based
    on mean connections."""

    if type(range_lower) != int or type(range_upper) != int:
        raise TypeError("upper and lower ranges must be an integers")
    if range_lower > range_upper:
        raise ValueError("lower range must be smaller than upper range")

    ave_k = 0
    prob_ks = {}

    for k in range(range_lower, range_upper+1):
        prob_ks[k] = k**(-exponent)

    normfactor = sum(prob_ks.values())

    for k in prob_ks:
        prob_ks[k] /= normfactor
        ave_k += k*prob_ks[k]

    if output == 0:
        return prob_ks
    elif output == 1:
        return ave_k
    else:
        return prob_ks, ave_k

def inv_cf(prob_ks):
    '''Takes a dictionary of {real:probability of real} and turns it into a dictionary of {cumulative probability: real}
    in the form of an inverse cumulative function.'''
    cumulative = 0
    inv_cfs = {}
    for key in sorted(list(prob_ks.keys())):
        cumulative += prob_ks[key]
        inv_cfs[cumulative] = key

    if abs(cumulative - 1) > 0.01:
        warnings.warn('The cumulative sum does not equal 1 instead it equals, check if a probability function has been given')
        print(f'The cumulative sum is {cumulative}')
    return inv_cfs

def sample_inv_cf(inv_cfs):
    '''Takes a dictionary of {cumulative probability: real} and samples from it using a uniform distribution.'''
    sorted_keys = sorted(list(inv_cfs.keys()))
    return inv_cfs[sorted_keys[bisect_right(sorted_keys, random.random())]]

def plaw_mean_max(mean, range_upper, range_lower=1, output=0):
    '''Uses scipy root_scaler to find the power law exponent value, and then calls plaw_exp_max to return the
     dictionary of {int: probability of int} for a discrete truncated power law distribution'''

    if type(range_lower) != int or type(range_upper) != int:
        raise TypeError("upper and lower ranges must be an integers")
    if range_lower > range_upper:
        raise ValueError("lower range must be smaller than upper range")
    if mean <= 0:
        raise ValueError("mean must be positive")

    def eq_to_solve(s):
        norm_factor = 0
        ave = 0
        for k in range(range_lower, range_upper+1):
            norm_factor += k**(-s)
            ave += k**(1-s)

        return ave/norm_factor - mean

    exponent = root_scalar(eq_to_solve, x0=2, method='secant').root

    return plaw_exp_max(exponent, range_upper, range_lower, output)

