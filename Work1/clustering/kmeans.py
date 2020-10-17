import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
"""
https://en.wikipedia.org/wiki/K-means_clustering
"""

def l2dist(points, center, axis=0):
    dist = (points - center)**2
    dist = np.sum(dist, axis=1)
    return np.sqrt(dist)

class KMeans:
    def __init__(self, n_clusters=8, *, distanceType='euclidean', max_iter=500,
                 tol=1e-4, verbose=False):
        self.numberOfClusters = n_clusters
        self.distanceType = distanceType
        self.maxIterations = int(max_iter)
        self.maxStopDistance = tol
        self.verbose = verbose
        self.centers = None
        self.accuracy = []

    def fit(self, trainData, y=None):
        trainData = self._convertToNumpy(trainData)
        self._initializeCenters(trainData)
        clusterLabels = np.random.randint(0, high=self.numberOfClusters, size=(trainData.shape[0],))
        for iterationIdx in range(self.maxIterations):
            previousLabels, previousCenters = clusterLabels, self.centers.copy()
            clusterLabels = self._predictClusters(trainData)
            self._updateCenters(trainData, clusterLabels)
            if self.verbose and y is not None:
                self.accuracy.append(np.sum(clusterLabels == y)/len(y))
            if self._stopIteration(previousCenters, self.centers, previousLabels, clusterLabels):
                break

    def predict(self, data):
        if self.centers is None:
            raise ValueError("Clusters have not been initialized, call KMeans.fit(X) first")
        return self._predictClusters(data)

    def fitPredict(self, data, y=None):
        self.fit(data, y=y)
        return self._predictClusters(data)

    def _initializeCenters(self, data):
        randomRowIdxs = np.random.choice(data.shape[0], self.numberOfClusters)
        self.centers = data[randomRowIdxs]
        if self.verbose:
            print("Initialization complete")

    def _convertToNumpy(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, list):
            data = np.array(data)
            if len(data) == 2:
                return data
        raise ValueError(f"type {type(data)} not supported")

    @staticmethod
    def _computeNewCenter(trainData, clusterLabels, clusterIdx, currentCenter):
        return np.mean(trainData[clusterLabels == clusterIdx], axis=0)

    def _updateCenters(self, trainData, clusterLabels):
        for clusterIdx, center in enumerate(self.centers):
            self.centers[clusterIdx] = self._computeNewCenter(trainData, clusterLabels, clusterIdx, center)

    def _predictClusters(self, data):
        l2distances = cdist(data, self.centers, self.distanceType)
        return np.argmax(l2distances, axis=1)

    def _stopIteration(self, previousCentroids, newCentroids, previousLabels, newLabels):
        return self._centroidsNotChanged(previousCentroids, newCentroids) or self._pointsInSameCluster(previousLabels, newLabels)

    def _centroidsNotChanged(self, previousCentroids, newCentroids):
        iterationDistance = np.sum(np.abs(newCentroids - previousCentroids))/previousCentroids.shape[1]
        if self.verbose:
            print(f"Centers have changed: {iterationDistance}")
        return iterationDistance < self.maxStopDistance

    def _pointsInSameCluster(self, previousLabels, newLabels):
        boolArray = previousLabels == newLabels
        if self.verbose:
            print(f"Classifications changed: {np.sum(previousLabels != newLabels)}/{len(previousLabels)}")
        return np.all(boolArray)

