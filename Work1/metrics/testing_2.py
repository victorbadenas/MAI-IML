from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from src.dataset import ArffFile
import matplotlib.pyplot as plt
from sklearn.metrics import cluster as clusteringMetrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from src.clustering import KMeans, BisectingKMeans, KMeansPP, FCM
from sklearn.cluster import KMeans as SKMeans
import pandas as pd
'''
https://medium.com/@rodrigodutcosky/creating-a-3d-scatter-plot-from-your-clustered-data-with-plotly-843c20b78799
'''

def purity_score(y_true, y_pred):
    contingency_matrix = clusteringMetrics.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def _computeInertia(self, data, dataLabels):
    """
    Inertia computation. The inertia is defined as the sum of the distance from each data
    point to the closest center (the center of the cluster where they are assigned).

    .. math::
        I = \sum_{\\forall point} d(point, cr)
    
    where cr is the closest center to the point.
    """
    inertia = 0.0
    for clusterIdx in range(self.numberOfClusters):
        clusterData = data[dataLabels == clusterIdx]
        clusterCenter = self.centers[clusterIdx]
        inertia += np.sum(l2norm(clusterData, clusterCenter))
    return inertia




def runDBSCAN(data):
    clustering = DBSCAN(n_jobs=-1, eps=.75)
    labels = clustering.fit_predict(data)
    return labels

def sklearnKMeans(data, numOfClusters):
    clustering = SKMeans(n_clusters=numOfClusters, init='random')
    labels = clustering.fit_predict(data)
    return labels

def ourKMeans(data, numOfClusters):
    clustering = KMeans(n_clusters=numOfClusters, verbose=False)
    labels = clustering.fitPredict(data)
    return labels

def kMeansPP(data, numOfClusters):
    clustering = KMeans(n_clusters=numOfClusters, init='k-means++', verbose=False)
    labels = clustering.fitPredict(data)
    return labels

'''
def bisectingKmeans(data, numOfClusters):
    clustering = BisectingKMeans(n_clusters=numOfClusters, init='random', verbose=False)
    labels = clustering.fitPredict(data)
    return labels
''' 
file_paths=["C:/Users/stick/Documents/Py_Archieve/IML/MAI-IML-master/Work1/datasets/vote.arff","C:/Users/stick/Documents/Py_Archieve/IML/MAI-IML-master/Work1/datasets/adult.arff","C:/Users/stick/Documents/Py_Archieve/IML/MAI-IML-master/Work1/datasets/waveform.arff"]
for files in file_paths:                
    arffFile = ArffFile("C:/Users/stick/Documents/Py_Archieve/IML/MAI-IML-master/Work1/datasets/vote.arff")
    unsupervisedFeatures = arffFile.getData().copy()

    labelColumn = unsupervisedFeatures.columns[-1]
    unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
    y = arffFile.getData()[labelColumn]

    dataNumpy = unsupervisedFeatures.to_numpy()

    pca = PCA(n_components=3)

    K=range(2,10)
    for k in K:
        for cluster_method in range(1,4):
            if cluster_method=1:
                clusters = sklearnKMeans(unsupervisedFeatures,k)
            elif cluster_method=2:
                clusters = ourKMeans(unsupervisedFeatures,k)
            elif cluster_method=3:
                clusters = kMeansPP(unsupervisedFeatures,k)
            else:
                #clusters = bisectingKmeans(unsupervisedFeatures,k)
            
            pcaData = pca.fit_transform(dataNumpy)

            metrics = {}

            metrics["davies_bouldin_score"] = clusteringMetrics.davies_bouldin_score(unsupervisedFeatures, clusters)
            metrics["adjusted_mutual_info_score"] = clusteringMetrics.adjusted_mutual_info_score(y, clusters)
            metrics["adjusted_rand_score"] = clusteringMetrics.adjusted_rand_score(y, clusters)
            metrics["completeness_score"] = clusteringMetrics.completeness_score(y, clusters)
            metrics["fowlkes_mallows_score"] = clusteringMetrics.fowlkes_mallows_score(y, clusters)
            metrics["homogeneity_score"] = clusteringMetrics.homogeneity_score(y, clusters)
            metrics["mutual_info_score"] = clusteringMetrics.mutual_info_score(y, clusters)
            metrics["normalized_mutual_info_score"] = clusteringMetrics.normalized_mutual_info_score(y, clusters)
            metrics["v_measure_score"] = clusteringMetrics.v_measure_score(y, clusters)
            metrics["purity_score"] = purity_score(y, clusters)

            pprint(metrics)

            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for label in set(clusters):
                subData = pcaData[clusters == label]
                ax.scatter(subData[:, 0], subData[:, 1], subData[:, 2], s=60, label=label)
                ax.view_init(30, 185)
            plt.legend()
            plt.show()
            
            