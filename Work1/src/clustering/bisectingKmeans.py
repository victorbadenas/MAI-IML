import numpy as np
from .kmeans import KMeans
from ..utils import convertToNumpy, l2norm
from scipy.spatial.distance import cdist

"""
Input: number of clusters K, data points x1,...,x_n
Output: K cluster centers, c_1,...,c_k 
1. Pick a cluster to split.
2. Find 2 sub-clusters using the basic k-Means algorithm (Bisecting step)
3. Repeat step 2, the bisecting step, for ITER times and take the split that produces the clustering with the highest overall similarity.
4. Repeat steps 1, 2 and 3 until the desired number of clusters is reached.
"""


class BisectingKMeans:
    def __init__(self, n_clusters=8, *, init='random', n_init=10, max_iter=500, tol=1e-4, verbose=False):
        self.numberOfClusters = n_clusters
        self.verbose = verbose
        self.kmeans = KMeans(n_clusters=2, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose)
        self.centers = None

    def fitPredict(self, trainData):
        self.fit(trainData)
        return self.predict(trainData)

    def fit(self, trainData):
        trainData = convertToNumpy(trainData)
        self._initializeClusters(trainData)
        for _ in range(self.numberOfClusters - 1):
            self.dataLabels = self._predictClusters(trainData)
            biggestIdx = self._getBiggestCluster(trainData)
            clusterData = trainData[self.dataLabels == biggestIdx]
            self.kmeans.reset()
            self.kmeans.fit(clusterData)
            np.delete(self.centers, biggestIdx)
            for center in self.kmeans.centers:
                self._updateCenters(center)
        return self

    def predict(self, data):
        return self._predictClusters(data)

    def _updateCenters(self, newCenter):
        self.centers = np.concatenate((self.centers, [newCenter]), axis=0)

    def _getBiggestCluster(self, trainData):
        clusterDistances = [self._computeClusterSize(trainData, clusterIdx) for clusterIdx in range(len(self.centers))]
        return np.argmax(clusterDistances)

    def _computeClusterSize(self, trainData, clusterIdx):
        clusterData = trainData[self.dataLabels == clusterIdx]
        distances = l2norm(clusterData, self.centers[clusterIdx])
        return np.max(distances)

    def _predictClusters(self, data):
        """
        Compute the distances from each of the data points in data to each of the centers.
        data is of shape (n_samples, n_features) and centers is of shape (n_clusters, n_features).
        It will result in a l2distances matrix of shape (n_samples, n_clusters) of which the
        argmin function will return a (n_samples,) vector with the cluster assignation.
        """
        l2distances = cdist(data, self.centers)
        return np.argmin(l2distances, axis=1)

    def _initializeClusters(self, trainData):
        randomRowIdxs = np.random.choice(trainData.shape[0], 1)
        self.centers = trainData[randomRowIdxs]

    def get_centroids(self):
        return self.centers
