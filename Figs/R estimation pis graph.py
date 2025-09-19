import matplotlib.pyplot as plt
import numpy as np
from math import exp

times = np.linspace(0.0000001, 10, 5*100)

beta = 1
lambd = 1
I0 = 1000
ave_k = 5

def q1(t):
    return beta/(beta+lambd)*(1-exp(-1*(beta+lambd)*t))

def q2(t):
    return beta**2/(beta+lambd)**2 * (1-exp(-1*(beta+lambd)*t)*(1+(beta+lambd)*t))

def q3(t):
    return beta**3/(beta+lambd)**3 * (1-exp(-1*(beta+lambd)*t)*(1+(beta+lambd)*t+(beta+lambd)/2*t**2))

def q4(t):
    return beta**4/(beta+lambd)**4 * (1-exp(-1*(beta+lambd)*t)*(1+(beta+lambd)*t+(beta+lambd)/2*t**2+(beta+lambd)/6*t**3))


# size1 = [I0*(ave_k-1)*q1(t) for t in times]
# size2 = [size1[t]*(ave_k-1)*q2(times[t])/q1(times[t]) for t in range(len(times))]
# size3 = [size2[t]*(ave_k-1)*q3(times[t])/q2(times[t]) for t in range(len(times))]
# size4 = [size3[t]*(ave_k-1)*q4(times[t])/q3(times[t]) for t in range(len(times))]

size1 = [q1(t) for t in times]
size2 = [q2(t) for t in times]
size3 = [q3(t) for t in times]
size4 = [q4(t) for t in times]

plt.plot(times, size1, color='r')
plt.plot(times, size2, color='b')
plt.plot(times, size3, color='g')
plt.plot(times, size4, color='y')
plt.show()





# def nm(x, tau):
#     return (beta+lambd)/(1-exp(-1*(beta+lambd)*tau)) * exp(-1*(beta+lambd)*x)
# tau = 10
# xs = np.linspace(0, tau, 200)
# pdf = [nm(x, tau) for x in xs]
# plt.plot(xs, pdf)
# plt.show()
#
