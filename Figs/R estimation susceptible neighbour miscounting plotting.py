import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'R estimation susceptible neighbour miscounting data.csv')

#applying the prediction function
def prop_prediction(row):
    return row['susceptible neighbours']/(row['N']*row['ave_k']*row['sample'])
df['Simple Prediction'] = df.apply(prop_prediction, axis=1)
def prop_prediction_scaled(row):
    return row['susceptible neighbours']/(row['S at time of sample']*row['ave_k']*row['sample'])
df['Scaled Prediction'] = df.apply(prop_prediction_scaled, axis=1)
def prop_pred_overcount_removed(row):
    return row['susceptible neighbours']/(row['N']*row['ave_k']*row['sample']-row['num 2s']-2*row['num 3s']-3*row['num 4s']-4*row['num 5 or more'])
df['Simple Prediction Overcount Removed'] = df.apply(prop_pred_overcount_removed, axis=1)
def prop_pred_scaled_overcount_removed(row):
    return row['susceptible neighbours']/(row['S at time of sample']*row['ave_k']*row['sample']-row['num 2s']-2*row['num 3s']-3*row['num 4s']-4*row['num 5 or more'])
df['Scaled Prediction Overcount Removed'] = df.apply(prop_pred_scaled_overcount_removed, axis=1)

#boxplots for different methods
# df_box = df.filter(['sample', 'Simple Prediction', 'Scaled Prediction', 'Simple Prediction Overcount Removed', 'Scaled Prediction Overcount Removed'])
# df_box = df_box.melt(id_vars=['sample'])
#
# p1 = sns.boxplot(df_box, x='sample', y='value', hue='variable', fill=False, whis=(1, 99), flierprops={'marker':'.'})
# p1.set(xlabel='Proportion of N Infected', ylabel='Proportion of predicted quantity')
# p1.set(xticklabels=['1%', '2.5%', '5%', '7.5', '10%', '12.5%', '15%'])
# plt.show()


df_matrix = df.filter(['sample', 'num 1s', 'num 2s', 'num 3s', 'num 4s', 'num 5 or more'])
df_matrix['sum'] = df_matrix[list(df_matrix.columns)[1:]].sum(axis=1)
df_matrix[df_matrix.columns.tolist()[1:-1]] = df_matrix[list(df_matrix.columns)[1:-1]].div(df_matrix['sum'], axis=0)
df_matrix = df_matrix[df_matrix.columns.tolist()[:-1]]
df_matrix = df_matrix.melt(id_vars=['sample'])
df_matrix = df_matrix.groupby(['sample', 'variable']).mean().reset_index()
df_matrix = df_matrix.pivot(columns='sample', index='variable', values='value').round(decimals=4)

p2 = sns.heatmap(df_matrix, annot=True, cmap='Blues', vmin=0, vmax=1)
p2.set(xticklabels=['1%', '2.5%', '5%', '7.5', '10%', '12.5%', '15%'],
       yticklabels=['1', '2', '3', '4', '5+'],
       xlabel='Proportion of N Infected',
       ylabel='Number of Infected Neighbours')
plt.show()
