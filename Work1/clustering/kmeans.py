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
    def __init__(self, n_clusters=8, *, distanceType='euclidean', init='random', max_iter=500,
                 tol=1e-4, verbose=False):
        self.numberOfClusters = n_clusters
        self.distanceType = distanceType
        self.maxIterations = int(max_iter)
        self.maxStopDistance = tol
        self.verbose = verbose
        self.init = init
        self.centers = None
        self.inertias_ = []

    def fit(self, trainData, y=None):
        trainData = self._convertToNumpy(trainData)
        self._initializeCenters(trainData)
        clusterLabels = np.random.randint(0, high=self.numberOfClusters, size=(trainData.shape[0],))
        for iterationIdx in range(self.maxIterations):
            previousLabels, previousCenters = clusterLabels, self.centers.copy()
            clusterLabels = self._predictClusters(trainData)
            self._updateCenters(trainData, clusterLabels)
            self.inertias_.append(self._computeInertia(trainData, clusterLabels))

            if self.verbose:
                print(f"Iteration {iterationIdx} with inertia {self.inertias_[-1]:.2f}")
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
        if self.init == 'random':
            randomRowIdxs = np.random.choice(data.shape[0], self.numberOfClusters)
            self.centers = data[randomRowIdxs]
        else:
            self.centers = data[:self.numberOfClusters]
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
        return np.argmin(l2distances, axis=1)

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

    def _computeInertia(self, data, dataLabels):
        inertia = 0.0
        for clusterIdx in range(self.numberOfClusters):
            clusterData = data[dataLabels == clusterIdx]
            clusterCenter = self.centers[clusterIdx]
            inertia += np.sum(l2dist(clusterData, clusterCenter))
        return inertia
