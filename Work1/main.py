import argparse
import warnings
import numpy as np
warnings.simplefilter(action='ignore')
from dataset import ArffFile
from pathlib import Path
from sklearn.cluster import DBSCAN
from clustering import KMeans, BisectingKMeans, KMeansPP
from collections import Counter

def parseArguments():
    parser = argparse.ArgumentParser()
    # p.e. we could use adult.arff as mixed, vote.arff as nominal and waveform.arff as numerical
    parser.add_argument("-f", "--arffFilesPaths", action='append', type=Path, required=True)
    return parser.parse_args()

class Main:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        print(self.args.arffFilesPaths)
        arffFile = ArffFile(self.args.arffFilesPaths[0])
        unsupervisedFeatures = arffFile.getData().copy().drop('class', axis=1)
        y = arffFile.getData()['class']
        print("number of classes=", len(np.unique(y)))
        print(unsupervisedFeatures.head())
        # arffFile.scatterPlot(figsize=(10, 10), ignoreLabel=True)
        clustering = DBSCAN(n_jobs=-1, eps=.75)
        labels = clustering.fit_predict(unsupervisedFeatures)
        print(np.unique(labels))
        print(Counter(labels))

if __name__ == "__main__":
    args = parseArguments()
    Main(args)()