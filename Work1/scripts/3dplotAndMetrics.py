import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import cluster as clusteringMetrics
from sklearn.metrics import silhouette_score
warnings.filterwarnings(action='ignore')

sys.path.append("..")
from src.clustering import KMeans, BisectingKMeans, FCM
from src.dataset import ArffFile
from src.metrics import clusteringMappingMetric, purity_score

def runDBSCAN(data, numOfClusters=None):
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

def bisectingKmeans(data, numOfClusters):
    clustering = BisectingKMeans(n_clusters=numOfClusters, init='random', verbose=False)
    labels = clustering.fitPredict(data)
    return labels

def fcm(data, numOfClusters):
    clustering = FCM(n_clusters=numOfClusters, verbose=False)
    labels = clustering.fitPredict(data)
    return labels


file_paths = [Path("../datasets/vote.arff"), Path("../datasets/adult.arff"), Path("../datasets/pen-based.arff")]
# file_paths = [Path("../datasets/adult.arff"), Path("../datasets/pen-based.arff")]
RESULTS_BASE = Path("../results")
clusterMethods = [sklearnKMeans, ourKMeans, kMeansPP, bisectingKmeans, fcm]
# clusterMethods = [runDBSCAN]
K = 20
pca = PCA(n_components=3)


for filePath in file_paths:
    print(f"processing {filePath}...")
    arffFile = ArffFile(filePath, missingDataImputation="median")
    unsupervisedFeatures = arffFile.getData().copy()

    labelColumn = unsupervisedFeatures.columns[-1]
    unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
    y = arffFile.getData()[labelColumn]

    dataNumpy = unsupervisedFeatures.to_numpy()

    for cluster_method in clusterMethods:
        print(f"{cluster_method.__name__} is being computed:")

        resultsDataframe = pd.DataFrame(index=["davies_bouldin_score", "adjusted_rand_score", "completeness_score", "purity_score"])
        results_dir = RESULTS_BASE / filePath.stem / cluster_method.__name__
        results_dir.mkdir(exist_ok=True, parents=True)
        dfPath = results_dir / "data"
        dfPath.mkdir(exist_ok=True)

        silhouettes = []
        for k in range(2, K):
            print(f"{k} number of clusters")
            clusters = cluster_method(unsupervisedFeatures, k)
            pcaData = pca.fit_transform(dataNumpy)

            metrics = dict()
            metrics["davies_bouldin_score"] = clusteringMetrics.davies_bouldin_score(unsupervisedFeatures, clusters)
            metrics["adjusted_rand_score"] = clusteringMetrics.adjusted_rand_score(y, clusters)
            metrics["completeness_score"] = clusteringMetrics.completeness_score(y, clusters)
            metrics["purity_score"] = purity_score(y, clusters)
            confusionMatrix = clusteringMappingMetric(clusters, y)

            kdf = pd.DataFrame.from_dict(metrics, orient='index', columns=[k])
            resultsDataframe = resultsDataframe.join(kdf)

            silhouettes.append(silhouette_score(unsupervisedFeatures, clusters))

            np.savetxt(dfPath / f"{k}clusters_confusion.csv", confusionMatrix, delimiter=",", fmt='%i')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for label in set(clusters):
                subData = pcaData[clusters == label]
                ax.scatter(subData[:, 0], subData[:, 1], subData[:, 2], s=10, label=label)
                ax.view_init(30, 185)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / f"3DScatter_{k}clusters.png")
            plt.close()

        plt.plot(list(range(2, K)), silhouettes)
        plt.xlabel("K")
        plt.ylabel("silhouette score")
        plt.savefig(results_dir / f"silhouettes.png")
        plt.close()

        resultsDataframe.to_csv(dfPath / "metric.csv")
