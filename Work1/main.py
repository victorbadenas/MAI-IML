import argparse
import warnings
import numpy as np
warnings.simplefilter(action='ignore')
from src.dataset import ArffFile
from pathlib import Path
from sklearn.cluster import DBSCAN
from src.clustering import KMeans, BisectingKMeans, KMeansPP
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
        arffFile = ArffFile(arffFilePath)
        unsupervisedFeatures = arffFile.getData().copy()
        if 'class' in unsupervisedFeatures:
            unsupervisedFeatures = unsupervisedFeatures.drop('class', axis=1)
            y = arffFile.getData()['class']
        elif 'Class' in unsupervisedFeatures:
            unsupervisedFeatures = unsupervisedFeatures.drop('Class', axis=1)
            y = arffFile.getData()['Class']
        else:
            #TODO Raises error for autos.arff
            raise ValueError
        numOfClasses = len(np.unique(y))
        _, slacc = self.sklearnKMeans(unsupervisedFeatures, y, numOfClasses)
        _, ouracc = self.ourKMeans(unsupervisedFeatures, y, numOfClasses)
        _, ppacc = self.KMeansPP(unsupervisedFeatures, y, numOfClasses)
        _, bkmacc = self.BisectingKmeans(unsupervisedFeatures, y, numOfClasses)
        return slacc, ouracc, ppacc, bkmacc


    @timer(print_=True)
    def sklearnKMeans(self, data, y, numOfClasses):
        clustering = SKMeans(n_clusters=numOfClasses, init='random')
        labels = clustering.fit_predict(data)
        acc = np.sum(labels == y)*100.0/len(labels)
        return labels, acc

    @timer(print_=True)
    def ourKMeans(self, data, y, numOfClasses):
        clustering = KMeans(n_clusters=numOfClasses, verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        acc = np.sum(labels == y)*100.0/len(labels)
        return labels, acc

    @timer(print_=True)
    def KMeansPP(self, data, y, numOfClasses):
        clustering = KMeans(n_clusters=numOfClasses, init='k-means++', verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        acc = np.sum(labels == y)*100.0/len(labels)
        return labels, acc

    @timer(print_=True)
    def BisectingKmeans(self, data, y, numOfClasses):
        clustering = BisectingKMeans(n_clusters=8, init='random', verbose=self.args.verbose)
        labels = clustering.fitPredict(data)
        acc = np.sum(labels == y)*100.0/len(labels)
        return labels, acc

    def runDBSCAN(self, data):
        clustering = DBSCAN(n_jobs=-1, eps=.75)
        labels = clustering.fit_predict(data)
        print(np.unique(labels))
        print(Counter(labels))

if __name__ == "__main__":
    args = parseArguments()
    report = Main(args)()
    pprint(report)