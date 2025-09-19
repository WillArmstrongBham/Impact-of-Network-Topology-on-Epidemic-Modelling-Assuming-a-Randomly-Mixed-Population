import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Distance Weighted Graph data.csv")


df_bh = df[['Actual Beta', 'Beta Hat Node Infections', 'Beta Hat Infection num sus nbrs', 'Beta Hat Infection num inf nbrs']].copy()
df_bh.columns = ['Actual Beta', r'Node Infections', r'IN Susceptible Neighbours', r'IN Infectious Neighbours']
df_bh = pd.melt(df_bh, id_vars=['Actual Beta'])
df_bh.columns = ['Actual Beta', 'Method of Estimation', 'value']

plt.rcParams.update({'text.usetex': True,
                     'font.family': 'serif',
                     'font.serif': 'Computer Modern',
                     'font.size': 14,
                     'figure.figsize':(6,6),
                    'axes.labelsize': 16,
                    'axes.titlesize':16})


p = sns.scatterplot(df_bh, x='Actual Beta', y='value', hue='Method of Estimation')
plt.plot([0, 10], [0, 10], color='grey', linestyle='--')
plt.xlabel(r'Actual $\beta$')
plt.ylabel(r'Predicted $\hat \beta$')
plt.title(r'$\hat \beta$ Estimation for SFLWN')
plt.xlim(0, 10)
plt.savefig("Beta Hat Estimation for SFLWN Graphs.pdf", bbox_inches='tight')
plt.close()

df_fc = df[['Actual Final Size', 'Beta Hat Node Infections Forecast', 'Beta Hat Infection num sus nbrs Forecast', 'Beta Hat Infection num inf nbrs Forecast']].copy()
df_fc.columns = ['Actual Final Size', 'Node Infections', 'IN Susceptible Neighbours', 'IN Infectious Neighbours']
df_fc = pd.melt(df_fc, id_vars='Actual Final Size')
df_fc.columns = ['Actual Final Size', 'Method of Estimation', 'value']
p = sns.scatterplot(df_fc, x='Actual Final Size', y='value', hue='Method of Estimation', legend=False)
plt.plot([0, 10000], [0, 10000], color='grey', linestyle='--')
plt.ylabel("Forecast Size")
plt.title(r"Forecast Sizes using Estimated $\hat \beta$ for SFLWN Graphs")
plt.savefig("Forecast Sizes for SFLWN Graphs.pdf", bbox_inches='tight')
plt.close()