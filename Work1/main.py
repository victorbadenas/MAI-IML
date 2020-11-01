import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import cluster as clusteringMetrics

warnings.simplefilter(action='ignore')
from src.utils import timer
from src.dataset import ArffFile
from src.clustering import KMeans, BisectingKMeans, FCM
from src.metrics import clusteringMappingMetric, purity_score

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--jsonConfig", type=Path, required=True)
    return parser.parse_args()

class Main:
    def __init__(self, args):
        self.args = args
        with open(self.args.jsonConfig , 'r') as f:
            self.config = json.load(f)

    def __call__(self):
        data, trueLabels = self.loadArffFile(Path(self.config["path"]))
        predictedLabels = self.runAlgorithms(data)
        pprint(predictedLabels)

        metrics, confusionMatrixes = self.computeMetrics(data, trueLabels, predictedLabels)
        self.saveMetrics(metrics,confusionMatrixes)
        if self.config["plotClusters"]:
            self.plotClusters(data, predictedLabels)

    def runAlgorithms(self, data):
        labels = {}
        if "DBSCAN" in self.config["parameters"]:
            labels["DBScan"] = self.runDBSCAN(data, **self.config["parameters"]["DBSCAN"])
        if "kMeans" in self.config["parameters"]:
            labels["ourKMeans"] = self.ourKMeans(data, **self.config["parameters"]["kMeans"])
        if "bisectingKMeans" in self.config["parameters"]:
            labels["bisectingKmeans"] = self.bisectingKmeans(data, **self.config["parameters"]["bisectingKMeans"])
        if "kmeansPP" in self.config["parameters"]:
            self.config["parameters"]["kmeansPP"]["init"] = "k-means++"
            labels["kMeansPP"] = self.ourKMeans(data, **self.config["parameters"]["kmeansPP"])
        if "fcm" in self.config["parameters"]:
            labels["fcm"] = self.fcm(data, **self.config["parameters"]["fcm"])
        return labels

    def computeMetrics(self, data, trueLabels, predictedLabels):
        confusionMatrixes = dict()
        metrics = dict()
        for algorithmName, labels in predictedLabels.items():
            metrics[algorithmName] = dict()
            metrics[algorithmName]["davies_bouldin_score"] = clusteringMetrics.davies_bouldin_score(data, labels)
            metrics[algorithmName]["adjusted_rand_score"] = clusteringMetrics.adjusted_rand_score(trueLabels, labels)
            metrics[algorithmName]["completeness_score"] = clusteringMetrics.completeness_score(trueLabels, labels)
            metrics[algorithmName]["purity_score"] = purity_score(trueLabels, labels)
            confusionMatrixes[algorithmName] = clusteringMappingMetric(labels, trueLabels)
        return metrics, confusionMatrixes

    def saveMetrics(self, metrics, confusionMatrixes):
        savePath = Path(self.config["resultsDir"]) / "main"
        savePath.mkdir(exist_ok=True, parents=True)
        pd.DataFrame.from_dict(metrics).to_csv(savePath / "metrics.csv")
        for algoName, matrix in confusionMatrixes.items():
            np.savetxt(savePath / f"{algoName}_confusion.csv", matrix, delimiter=",", fmt='%i')
        with open(savePath / "config.json", 'w') as f:
            json.dump(self.config, f)

    def plotClusters(self, data, predictedLabels):
        savePath = Path(self.config["resultsDir"]) / "main"
        pcaData = PCA(n_components=3).fit_transform(data)
        for algorithmName, labels in predictedLabels.items():
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for label in set(labels):
                subData = pcaData[labels == label]
                ax.scatter(subData[:, 0], subData[:, 1], subData[:, 2], s=10, label=label)
                ax.view_init(35, 145)
            plt.legend()
            plt.savefig(savePath / f"3DScatter_{algorithmName}.png")
            plt.close()

    @staticmethod
    def loadArffFile(arffFilePath):
        # load arff files
        arffFile = ArffFile(arffFilePath)
        unsupervisedFeatures = arffFile.getData().copy()

        # remove label column from training data
        labelColumn = unsupervisedFeatures.columns[-1]
        unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
        y = arffFile.getData()[labelColumn]

        return unsupervisedFeatures, y

    @timer(print_=True)
    def ourKMeans(self, data, **kwargs):
        clustering = KMeans(**kwargs)
        labels = clustering.fitPredict(data)
        return labels

    @timer(print_=True)
    def kMeansPP(self, data, **kwargs):
        clustering = KMeans(**kwargs)
        labels = clustering.fitPredict(data)
        return labels

    @timer(print_=True)
    def bisectingKmeans(self, data, **kwargs):
        clustering = BisectingKMeans(**kwargs)
        labels = clustering.fitPredict(data)
        return labels

    @timer(print_=True)
    def fcm(self, data, **kwargs):
        clustering = FCM(**kwargs)
        labels = clustering.fitPredict(data)
        return labels

    def runDBSCAN(self, data, **kwargs):
        clustering = DBSCAN(**kwargs)
        labels = clustering.fit_predict(data)
        return labels

if __name__ == "__main__":
    args = parseArguments()
    Main(args)()
