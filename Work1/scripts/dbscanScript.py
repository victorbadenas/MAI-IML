import sys
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import cluster as clusteringMetrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

sys.path.append("..")
from src.dataset import ArffFile
from src.metrics import clusteringMappingMetric

def runDBSCAN(data, **kwargs):
    clustering = DBSCAN(**kwargs)
    return clustering.fit_predict(data)


def main():
    filePaths = [Path("../datasets/vote.arff"), Path("../datasets/adult.arff"), Path("../datasets/pen-based.arff")]

    filePath = filePaths[1]

    arffFile = ArffFile(filePath)
    unsupervisedFeatures = arffFile.getData().copy()

    labelColumn = unsupervisedFeatures.columns[-1]
    unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
    trueLabels = arffFile.getData()[labelColumn]

    # dataNumpy = unsupervisedFeatures.to_numpy()
    eps = .5
    min_samples = range(10, 200, 10)
    n_jobs = -1

    for value in min_samples:
        print(f"Evaluating DBSCAN with parameter={value}")
        labels = runDBSCAN(unsupervisedFeatures, eps=eps, min_samples=value, n_jobs=n_jobs)
        extendedConfusion = confusion_matrix(trueLabels, labels)
        csvFilePath = Path(f"tmp/{filePath.stem}/")
        csvFilePath.mkdir(exist_ok=True, parents=True)
        np.savetxt(csvFilePath / f"{eps}eps_{value}min_samples_confusion.csv", extendedConfusion, delimiter=",", fmt='%i')


def findEps():
    from sklearn.neighbors import NearestNeighbors
    import seaborn as sns
    sns.set()

    filePaths = [Path("../datasets/vote.arff"), Path("../datasets/adult.arff"), Path("../datasets/pen-based.arff")]

    filePath = filePaths[2]

    arffFile = ArffFile(filePath)
    unsupervisedFeatures = arffFile.getData().copy()

    labelColumn = unsupervisedFeatures.columns[-1]
    unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
    _ = arffFile.getData()[labelColumn]

    X = unsupervisedFeatures.to_numpy()

    nn = 2
    neigh = NearestNeighbors(n_neighbors=nn)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    plt.plot(distances)
    plt.tight_layout()
    plotPath = Path(f"../results/{filePath.stem}/dbscan")
    plotPath.mkdir(exist_ok=True, parents=True)
    plt.savefig(plotPath / f"dbscan_{nn}.png")
    plt.show()


if __name__ == "__main__":
    main()