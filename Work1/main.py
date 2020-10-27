import argparse
import warnings
import numpy as np
warnings.simplefilter(action='ignore')
from src.dataset import ArffFile
from pathlib import Path
from sklearn.cluster import DBSCAN
from src.clustering import KMeans, BisectingKMeans, KMeansPP, FCM
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKMeans
from src.utils import timer
from pprint import pprint

def parseArguments():
    parser = argparse.ArgumentParser()
    # p.e. we could use adult.arff as mixed, vote.arff as nominal and waveform.arff as numerical
    parser.add_argument("-f", "--arffFilesPaths", action='append', type=Path, required=True)
    parser.add_argument("-v", "--verbose", action='store_true', default=False)
    return parser.parse_args()

class Main:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        report = {}
        for arffFilePath in self.args.arffFilesPaths:
            accs = self.runSingleFile(arffFilePath)
            report[str(arffFilePath)] = accs
        return report

    def runSingleFile(self, arffFilePath):
        # load arff files
        arffFile = ArffFile(arffFilePath)
        unsupervisedFeatures = arffFile.getData().copy()

        # remove label column from training data
        labelColumn = unsupervisedFeatures.columns[-1]
        unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
        y = arffFile.getData()[labelColumn]

        numOfClasses = len(np.unique(y))
        labels = {}
        labels["DBScan"] = self.runDBSCAN(unsupervisedFeatures)
        labels["sklearnKMeans"] = self.sklearnKMeans(unsupervisedFeatures, numOfClasses)
        labels["ourKMeans"] = self.ourKMeans(unsupervisedFeatures, numOfClasses)
        labels["kMeansPP"] = self.kMeansPP(unsupervisedFeatures, numOfClasses)
        labels["bisectingKmeans"] = self.bisectingKmeans(unsupervisedFeatures, numOfClasses)
        labels["fcm"] = self.fcm(unsupervisedFeatures, numOfClasses)
        return labels

    @timer(print_=True)
    def sklearnKMeans(self, data, numOfClusters):
        clustering = SKMeans(n_clusters=numOfClusters, init='random')
        labels = clustering.fit_predict(data)
        return labels

    @timer(print_=True)
    def ourKMeans(self, data, numOfClusters):
        clustering = KMeans(n_clusters=numOfClusters, verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        return labels

    @timer(print_=True)
    def kMeansPP(self, data, numOfClusters):
        clustering = KMeans(n_clusters=numOfClusters, init='k-means++', verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        return labels

    @timer(print_=True)
    def bisectingKmeans(self, data, numOfClusters):
        clustering = BisectingKMeans(n_clusters=numOfClusters, init='random', verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        return labels

    @timer(print_=True)
    def fcm(self, data, numOfClusters):
        clustering = FCM(n_clusters=numOfClusters, verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        return labels

    def runDBSCAN(self, data):
        clustering = DBSCAN(n_jobs=-1, eps=.75)
        labels = clustering.fit_predict(data)
        return labels

if __name__ == "__main__":
    args = parseArguments()
    report = Main(args)()
    pprint(report)
