import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import axes_style, plotting_context

df = pd.read_csv('Estimating Beta from infection numbers.csv')
df.columns = ['Actual Beta', 'From Susceptible Neighbours', 'From Infected Neighbours']
df = df.melt(id_vars=['Actual Beta'])
df.columns = ['Actual Beta', 'Estimation Method', 'value']

plt.rcParams.update({'text.usetex': True,
                     'font.family': 'serif',
                     'font.serif': 'Computer Modern',
                     'font.size': 14,
                     'figure.figsize':(6,6),
                    'axes.labelsize': 16,
                    'axes.titlesize':16})


ax = sns.scatterplot(df, x='Actual Beta', y='value', hue='Estimation Method')
plt.plot([0, 6], [0, 6], color='grey', linestyle='--')
ax.set_title(r"A plot of $\hat \beta$ against actual $\beta$")
ax.set_xlabel(r"Actual $\beta$")
ax.set_ylabel(r"Predicted $\hat \beta$")
plt.savefig('Estimating Beta from infection numbers.pdf', bbox_inches='tight')
