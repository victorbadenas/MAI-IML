import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKMeans

sys.path.append("..")
warnings.filterwarnings(action='ignore')

from src.clustering import KMeans, BisectingKMeans, FCM
from src.utils import timer
from src.dataset import ArffFile

@timer(print_=False)
def sklearnKMeans(data, numOfClusters):
    clustering = SKMeans(n_clusters=numOfClusters, init='random')
    labels = clustering.fit_predict(data)
    return labels

@timer(print_=False)
def ourKMeans(data, numOfClusters):
    clustering = KMeans(n_clusters=numOfClusters, verbose=False)
    labels = clustering.fitPredict(data)
    return labels

@timer(print_=False)
def kMeansPP(data, numOfClusters):
    clustering = KMeans(n_clusters=numOfClusters, init='k-means++', verbose=False)
    labels = clustering.fitPredict(data)
    return labels

@timer(print_=False)
def bisectingKmeans(data, numOfClusters):
    clustering = BisectingKMeans(n_clusters=numOfClusters, init='random', verbose=False)
    labels = clustering.fitPredict(data)
    return labels

@timer(print_=False)
def fcm(data, numOfClusters):
    clustering = FCM(n_clusters=numOfClusters, verbose=False)
    labels = clustering.fitPredict(data)
    return labels

PLOTS_PATH = Path("../results/performance")
FILES_PATH = [Path("../datasets/vote.arff"), Path("../datasets/adult.arff"), Path("../datasets/pen-based.arff")]
ALGORITHMS = [sklearnKMeans, ourKMeans, kMeansPP, bisectingKmeans, fcm]
ALGORITHMS_NAMES = ["sklearnKMeans", "ourKMeans", "kMeansPP", "bisectingKmeans", "fcm"]
N_RUNS = 100
MAX_K = 20

def loadArffFile(filePath):
    arffFile = ArffFile(filePath)
    unsupervisedFeatures = arffFile.getData().copy()

    labelColumn = unsupervisedFeatures.columns[-1]
    unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
    y = arffFile.getData()[labelColumn]
    return unsupervisedFeatures, y

def bruteForceRuntime():
    for filePath in FILES_PATH:
        print(f"processing {filePath} ...")
        data, _ = loadArffFile(filePath)

        resultsFolder = PLOTS_PATH / "bruteforce" / filePath.stem
        resultsFolder.mkdir(exist_ok=True, parents=True)

        metrics = {}
        for algorithm, name in zip(ALGORITHMS, ALGORITHMS_NAMES):
            print(f"running {name}")
            times = list()
            for _ in range(N_RUNS):
                _, time = algorithm(data, 3)
                times.append(time)
            hist, binEdges = np.histogram(times, bins=20)
            binCenters = binEdges[:-1] + np.diff(binEdges)/2
            plt.stem(binCenters, hist)
            plt.ylabel("number of currences")
            plt.xlabel("execution time in seconds")
            plt.tight_layout()
            plt.savefig(resultsFolder / f"{name}.png")
            plt.close()
            metrics[name] = {"nobs":N_RUNS, "mean":np.mean(times), "var":np.std(times)}
            print(f"Mean: {metrics[name]['mean']}, Std: {metrics[name]['var']}")
        pd.DataFrame.from_dict(metrics).T.to_csv(resultsFolder / "metrics.csv")

def clusterONotation():
    filePath = FILES_PATH[0]
    data, labels = loadArffFile(filePath)
    resultsFolder = PLOTS_PATH / "ONotationCluster" / filePath.stem
    resultsFolder.mkdir(exist_ok=True, parents=True)

    for algorithm in ALGORITHMS:
        times = [None]*(MAX_K-1)
        kValues = list(range(1, MAX_K))
        for i, k in enumerate(kValues):
            _, time = algorithm(data, k)
            times[i] = time
        plt.plot(kValues, times)
        plt.tight_layout()
        plt.savefig(resultsFolder / f"{algorithm.__name__}.png")

if __name__ == "__main__":
    bruteForceRuntime()
