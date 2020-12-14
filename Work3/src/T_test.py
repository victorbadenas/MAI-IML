import numpy as np
from scipy import stats
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.tools import FigureFactory as FF
import matplotlib.pyplot as plt
import scipy


def stats_features(data1, data2):
    pop_1 = len(data1)
    pop_2 = len(data2)
    mean_1 = np.mean(data1)
    mean_2 = np.mean(data2)
    stdev1 = np.std(data1)
    stdev2 = np.std(data2)
    var1 = data1.var(ddof=1)
    var2 = data2.var(ddof=1)
    return pop_1, pop_2, mean_1, mean_2, stdev1, stdev2, var1, var2


def sample_size_calc(pop_size, stdev):
    # Confidence interval 95%
    # alpha 0.05
    error_margin = 0.05
    # z score for alpha
    z_score = 1.96
    sample_size_pt1 = (((z_score**2)*stdev*(1-stdev))/error_margin**2)
    sample_size_pt2 = (1+(((z_score**2)*stdev*(1-stdev))/(error_margin**2)*pop_size))
    sample_size = sample_size_pt1/sample_size_pt2
    return sample_size


def sample_size_global(pop_size1, stdev1, pop_size2, stdev2):
    sam1 = sample_size_calc(pop_size1, stdev1)
    sam2 = sample_size_calc(pop_size2, stdev2)
    sample_global = max(sam1, sam2)
    return sample_global


# data1 as to be accuracy array, data2 as to be executuion time
def bell_plot(data1, data2):
    max_data1 = np.max(data1)
    max_data2 = np.max(data2)
    min_data1 = np.min(data1)
    min_data2 = np.min(data2)
    data_range = (min(min_data1, min_data2), max(max_data1, max_data2))
    x = np.linspace(*data_range, 1000)
    y1 = stats.norm(loc=np.mean(data1), scale=np.std(data1))
    y2 = stats.norm(loc=np.mean(data2), scale=np.std(data2))
    plt.subplot(211)
    trace1 = plt.plot(x, y1.pdf(x), label='Mean of Accuraccy', c='b')
    plt.legend()
    plt.grid('on')
    plt.subplot(212)
    trace2 = plt.plot(x, y2.pdf(x), label='Mean of Time to Execute', c='r')
    plt.legend()
    plt.grid('on')
    plot_data = [trace1, trace2]
    return plot_data


# Run T-test function
# Output t-test, p-value
# T-test = ratio between the difference between two groups and the difference within the groups
# P-value = measure of the probability that an observed difference could have occurred just by random chance
def t_test(data1, data2):
    t, p = stats.ttest_ind(data1, data2)
    # Create Bell plot
    bell_plot(data1, data2)
    plt.show()
    # Table with T-Test results
    table_matrix = [['Sample Data', t, p]]
    # t_table = FF.create_table(table_matrix, index=True)
    df = pd.DataFrame(table_matrix, columns=['T-Test', 'Test Statistic', 'p-value']).set_index(['T-Test'])
    return df


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("results/best_knn/autos.tsv", sep='\t', index_col=0)
    df = t_test(data['mean_test_score'], data['mean_score_time'])
    print(df)
