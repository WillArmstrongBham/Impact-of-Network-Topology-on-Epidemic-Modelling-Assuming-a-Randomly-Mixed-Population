from math import factorial
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd

N = 9925
ave_k = 5
upper_limit = 20

def i_nbrs_given_k(i, k, I0, N):
    prob = 1
    for j in range(k):
        if j < i:
            prob *= I0-j
        if j < k-i:
            prob *= N-1-I0-j
        prob /= N-1-j
    prob *= factorial(k)/factorial(i)/factorial(k-i)
    return prob

def prob_infect(I0, N):
    prob_infect = [0]*upper_limit
    for num_nbrs in range(1, upper_limit):
        for i in range(num_nbrs+1):
            prob_infect[i] += stats.binom.pmf(num_nbrs, N, ave_k/N)*i_nbrs_given_k(i, num_nbrs, I0, N)
    return prob_infect

def average_prob_infect(I0, N):
    prob_infect = [0]*upper_limit
    for i in range(ave_k+1):
        prob_infect[i] += i_nbrs_given_k(i, ave_k, I0, N)
    return prob_infect

# ave_is = [0]*upper_limit
# for i in range(ave_k+1):
#     ave_is[i] = i_nbrs_given_k(i, ave_k)
#
# prob_infect = [0]*upper_limit
# for num_nbrs in range(1, upper_limit):
#     for i in range(num_nbrs+1):
#         prob_infect[i] += stats.binom.pmf(num_nbrs, N, ave_k/N)*i_nbrs_given_k(i, num_nbrs)
#
# num_infected_neibours = list(range(upper_limit))
# df = pd.DataFrame([num_infected_neibours, ave_is, prob_infect])
# df = df.transpose()
# df.columns = ['Number of Infected Vertices', 'Assuming average degree', 'Conditioning on degree']
# df= df.head(11)
# df = df.melt(id_vars=['Number of Infected Vertices'])
# sns.barplot(df, x='Number of Infected Vertices', y='value', hue='variable')
# plt.show()

df = pd.read_csv(r'R estimation I nbrs condition data.csv')

plt.rcParams.update({'text.usetex': True,
                     'font.family': 'serif',
                     'font.serif': 'Computer Modern',
                     'font.size': 14,
                     'figure.figsize':(8,6),
                    'axes.labelsize': 16,
                    'axes.titlesize':16})


fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1,3]})
counter = 0
for sample in [0.01, 0.25]:
    #restrict df to just sample
    temp_df = df[df['Sample']==sample].copy()
    #from each value of N and I predict the distribution and then average the results - one for conditional prediction one for average
    simulated_dist = temp_df[['N', 'I']].copy().apply(lambda row: pd.Series(prob_infect(row['I'], row['N'])), axis=1).mean().to_frame().T
    simulated_ave = temp_df[['N', 'I']].copy().apply(lambda row: pd.Series(average_prob_infect(row['I'], row['N'])), axis=1).mean().to_frame().T
    #turn actual data into a probability distribution then average
    actual_dist = temp_df[temp_df.columns.values[4:]].div(temp_df['S'], axis=0).mean().copy().to_frame().T
    #wrangle to get into plottable form and then plot
    simulated_dist.columns = actual_dist.columns.values.tolist()
    simulated_ave.columns = actual_dist.columns.values.tolist()
    plot_df = pd.concat([actual_dist, simulated_dist, simulated_ave], axis=0)
    plot_df['Source'] = ['Simulated Data', 'Estimate with Conditioning', 'Estimate with Average Degree']
    plot_df = plot_df[ plot_df.columns.values.tolist()[0:6] +['Source']]
    plot_df = plot_df.melt(id_vars=['Source'])
    plot_df.columns = ['Source', 'Number of Infected Neighbours', 'Probability']

    #colour_palette = ['#0080FF', '#', '#']
    p = sns.barplot(plot_df, x='Number of Infected Neighbours', y='Probability', hue='Source', ax=axs[counter])
    p.legend_.remove()
    p.set(xlabel=None, ylabel=None)
    p.set(ylim=(0, 1))
    p.grid(True, axis='y')
    sample = int(sample*100)
    p.set_title(str(sample)+r"\% of $N$ infected")
    if counter == 0:
        p.set(xlim=(-0.5,1.5))
    if counter == 1:
        p.set(yticklabels=[])
        for tick in p.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
    counter += 1

plt.legend(framealpha=1, edgecolor='grey')
fig.suptitle("Probability Density Distribution of Number of Infected Neighbours ")
fig.supxlabel('Number of Infected Neighbours')
fig.supylabel('Probability')
fig.tight_layout()

plt.savefig('R estimation I nbrs condition.pdf', bbox_inches='tight')
