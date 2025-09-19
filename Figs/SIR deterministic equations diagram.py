import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

N = 100
beta = 1.5
gamma = 0.3

def returns_SIRdt(SIR, t):
    S, I, R = SIR
    dS = -1*beta/N*S*I
    dI = beta/N*S*I - gamma*I
    dR = gamma*I
    return [dS, dI, dR]

S1 = 99
I1 = 1
R1 = 0
t = np.linspace(0, 15, 400)
sol = odeint(returns_SIRdt, [S1, I1, R1], t)

S = sol[:,0]
I = sol[:,1]
R = sol[:,2]

plt.plot(t, S, color='g', label='Susceptible', marker='o', markevery=15)
plt.plot(t, I, color='r', label='Infectious', marker = '^', markevery=15)
plt.plot(t, R, color='b' ,label='Removed', marker='s', markevery=15)
plt.legend()
plt.suptitle("Simple SIR model")
plt.title(r"$\beta=1.5, \gamma=0.3$")
plt.savefig("SIR_diagram.png")
