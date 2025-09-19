import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for p in range(1, 8):
    y = p*x - (1 - (1 - x) ** p)  # initial error - error accounting for neighbours
    plt.plot(x, y, color='grey')
    ax.annotate(p, (1, p-1), xytext=(1.02, p-1), color='black')

plt.title("n*p_{1}")
plt.show()

