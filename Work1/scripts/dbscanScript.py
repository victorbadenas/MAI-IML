import sys
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

sys.path.append(".")
from src.dataset import ArffFile

def runDBSCAN(data, **kwargs):
    clustering = DBSCAN(**kwargs)
    return clustering.fit_predict(data)


def evaluate(data, y, eps, min_samples):
    n_jobs = -1

    for value in min_samples:
        for ep in eps:
            print(f"Evaluating DBSCAN with parameter={value}")
            labels = runDBSCAN(data, eps=ep, min_samples=value, n_jobs=n_jobs)
            extendedConfusion = confusion_matrix(y, labels)
            np.savetxt(f"results/adult/dbscan/{ep}eps_{value}min_samples_confusion.csv", extendedConfusion, delimiter=",", fmt='%i')


def findEps(data, Ks=[2]):
    X = data.to_numpy()
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(Ks))]
    for K, col in zip(Ks, colors):
        neigh = NearestNeighbors(n_neighbors=K+1)
        distances, _ = neigh.fit(X).kneighbors(X)
        distances = np.sort(distances[:, -1], axis=0)
        plt.plot(distances, color=tuple(col), label=f'{K} Nearest Neighbor')
    plt.tight_layout()
    plt.xlabel('Points sorted according to distance of Kth Nearest Neighbor')
    plt.ylabel('Kth Nearest Neighbor Distance')
    plt.title("K-dist plot for 'adult' dataset")
    plt.legend()
    plt.savefig(f"results/adult/dbscan/dbscanDistanceToKthNeighbor.png")
    plt.show()


if __name__ == "__main__":
    arffFile = ArffFile(Path("./datasets/adult.arff"))
    data = arffFile.getData().copy()

    labelColumn = data.columns[-1]
    y = data[labelColumn]
    data = data.drop(labelColumn, axis=1)

    minPts = [10, 20, 30, 40, 50]
    eps = [.55, .6, .65, .70, .75]

    findEps(data, Ks=minPts)
    #evaluate(data, y, eps, minPts)