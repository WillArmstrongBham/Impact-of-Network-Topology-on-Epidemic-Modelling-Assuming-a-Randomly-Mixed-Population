from numpy.random import exponential
import matplotlib.pyplot as plt

zs = []
beta = 1/10
gamma = 1
tau = 2

for a in range(1000000):
    x0, x1 = exponential(beta, 2)
    y0, y1 = exponential(gamma, 2)
    if x0 < y0 and x1 < y1 and x0+x1 < tau:
        zs.append(x0+x1)

plt.hist(zs, bins =50)
plt.show()