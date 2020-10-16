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
    def __init__(self, numberOfClusters, distanceType='euclidean', maxIterations=500):
        self.numberOfClusters = numberOfClusters
        self.distanceType = distanceType
        self.maxIterations = int(maxIterations)
        self.maxStopDistance = 1e-9
        self.centers = None
        self.accuracy = []

    def fit(self, trainData, y=None):
        trainData = self._convertToNumpy(trainData)
        self._initializeCenters(trainData)
        clusterLabels = np.random.randint(0, high=self.numberOfClusters, size=(trainData.shape[0],))
        for iterationIdx in tqdm.tqdm(range(self.maxIterations)):
            previousLabels, previousCenters = clusterLabels, self.centers.copy()
            clusterLabels = self._predictClusters(trainData)
            self._updateCenters(trainData, clusterLabels)
            if y is not None:
                self.accuracy.append(np.sum(clusterLabels == y)/len(y))
            if self._stopIteration(previousCenters, self.centers, previousLabels, clusterLabels):
                break

    def predict(self, data):
        if self.centers is None:
            raise ValueError("Clusters have not been initialized, call KMeans.fit(X) first")
        return self._predictClusters(data)

    def fitPredict(self, data, y=None):
        self.fit(data, y=y)
        return self.predict(data)

    def _initializeCenters(self, data):
        randomRowIdxs = np.random.choice(data.shape[0], self.numberOfClusters)
        self.centers = data[randomRowIdxs]

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
    def _computeNewCenter(trainData, clusterTags, clusterIdx, currentCenter):
        return np.mean(trainData[clusterTags == clusterIdx], axis=0)

    def _updateCenters(self, trainData, clusterTags):
        for clusterIdx, center in enumerate(self.centers):
            self.centers[clusterIdx] = self._computeNewCenter(trainData, clusterTags, clusterIdx, center)

    def _predictClusters(self, data):
        l2distances = cdist(data, self.centers, self.distanceType)
        return np.argmax(l2distances, axis=1)

    def _stopIteration(self, previousCentroids, newCentroids, previousTags, newTags):
        return self._centroidsNotChange(previousCentroids, newCentroids) or self._pointsInSameCluster(previousTags, newTags)

    def _centroidsNotChange(self, previousCentroids, newCentroids):
        return np.sum(np.abs(newCentroids - previousCentroids))/previousCentroids.shape[1] < self.maxStopDistance

    @staticmethod
    def _pointsInSameCluster(previousTags, newTags):
        return np.all(previousTags == newTags)
