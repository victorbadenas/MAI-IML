import numpy as np
from scipy import stats
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import scipy

class T_test:
    def __init__(self, data1, data2):
        self.data1=data1
        self.data2=data2
    
    def stats_features(data1, data2):
        pop_1=len(data1)
        pop_2=len(data2)
        mean_1=np.mean(data1)
        mean_2=np.mean(data2)
        stdev1=np.std(data1)
        stdev2=np.std(data2)
        var1=data1.var(ddof=1)
        var2=data2.var(ddof=1)
        return pop_1, pop_2, mean_1, mean_2, stdev1, stdev2, var1, var2
        
    def sample_size_calc(pop_size, stdev):
        #Confidence interval 95%
        #alpha 0.05
        error_margin=0.05
        #z score for alpha
        z_score=1.96
        sample_size_pt1=(((z_score**2)*stdev(1-stdev))/error_margin**2)
        sample_size_pt2=(1+(((z_score**2)*stdev(1-stdev))/(error_margin**2)*pop_size))
        sample_size=sample_size_pt1/sample_size_pt2
        return sample_size

    def sample_size_global(pop_size1, stdev1, pop_size2, stdev2):
        sam1=sample_size_calc(pop_size1, stdev1)
        sam2=sample_size_calc(pop_size2, stdev2)
        sample_global=max(sam1,sam2)
        return sample_global

    #degree if freedom
    def t_test_run(mean_1, mean_2, var1, var2, n):
        stdev=np.sqrt((var1+var2)/2)
        t_test=(mean_1-mean_2)/(stdev*np.sqrt(2/n))
        p_value=1-stats.t.dist.cdv(t_test, df=df)
        return t_test, p_value

    #Function to create boxplots by providing a dataset range
    def create_boxplot(data):
        fig=plt.figure(figuresize=(10, 7))
        plt.boxplot(data)
        plt.show()
        
    #data1 as to be accuracy array, data2 as to be executuion time
    def bell_plot(data1, data2):
        max_data1=np.max(data1)
        max_data2=np.max(data2)
        min_data1=np.min(data1)
        min_data2=np.min(data2)
        x = np.linspace(min(max_data1, max_data2), max(min_data1, min_data2), 160)
        y1 = scipy.stats.norm.pdf(x)
        y2 = scipy.stats.norm.pdf(x, loc=2)
        trace1 = go.Scatter(x = x,y = y1,mode = 'lines+markers',name='Mean of Accuraccy')
        trace2 = go.Scatter(x = x,y = y2,mode = 'lines+markers',name='Mean of Time to Execute')
        plot_data = [trace1, trace2]
        return plot_data


    #Run T-test function    
    #Output t-test, p-value
    #T-test = ratio between the difference between two groups and the difference within the groups
    #P-value = measure of the probability that an observed difference could have occurred just by random chance
    def t_test(data1, data2):
        features=stats_features(data1, data2)
        #output of features: pop_1, pop_2, mean_1, mean_2, stdev1, stdev2, var1, var2
        n=sample_size_global(features[0], features[4], features[1], features[5])
        run=t_test_run(features[2], features[3], features[6], features[7], n)
        #Create Bell plot
        bell_plot=bell_plot(data1, data2)
        py.iplot(bell_plot, filename='norm_dist_plot')
        #Table with T-Test results
        table_matrix = [['T-Test', 'Test Statistic', 'p-value'],['Sample Data', run[0], run[1]]]
        t_table = FF.create_table(table_matrix, index=True)
        py.iplot(t_table, filename='t-table')
