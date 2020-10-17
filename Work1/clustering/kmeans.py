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
    """KMeans Clustering algorithm:

    This object is responsible of performing the kmeans algorithm in a
    set of data and compute its centers.

    Parameters:
        n_clusters : int, default=8
            The number of clusters to form as well as the number of
            centroids to generate.

        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a
            single run.

        init : {'random', 'first'}, default='random'
            Method for initialization:

            'random': choose `n_clusters` observations (rows) at random from data
            for the initial centroids.

            'first': choose the first `n_clusters` observations (rows) from data
            for the initial centroids.

        tol : float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two consecutive iterations to declare
            convergence.

        verbose : bool, default=False
            Verbosity mode.

    """
    def __init__(self, n_clusters=8, *, init='random', max_iter=500, tol=1e-4, verbose=False):
        self.numberOfClusters = n_clusters
        self.maxIterations = int(max_iter)
        self.maxStopDistance = tol
        self.verbose = verbose
        self.init = init
        self.centers = None
        self.inertias_ = []

    """
    Main call functions for the algorithm:
    - fit: compute centers adequate to a dataset X
    - predict: compute cluster indexes for a test dataset
    - fitPredict: realizes the two functions above for the same dataset and returns the labels
    """

    def fit(self, trainData):
        """Compute kmeans centroids for trainData.
        
        Parameters:
            trainData: {np.ndarray, pd.DataFrame, list} of shape (n_samples, n_features)
                Training instances to compute cluster centers

        Returns:
            self: fitted estimator
        """
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
        return self

    def predict(self, data):
        """Compute kmeans labels for trainData given previously computed centroids.
        
        Parameters:
            data: {np.ndarray, pd.DataFrame, list} of shape (n_samples, n_features)
                Training instances to infer.

        Returns:
            labels: np.ndarray of shape (n_samples,) containing int data with the cluster
                index for each sample in data

        Notes:
            n_features of data must match n_feaures of self.centers for correctly 
            computing the labels, otherwise `ValueError` will be raised.
        """
        if self.centers is None:
            raise ValueError("Clusters have not been initialized, call KMeans.fit(X) first")
        data = self._convertToNumpy(data)
        if self.centers.shape[1] != data.shape[1]:
            raise ValueError(f"{data.shape} is an invalid data structure for kmeans of centers {self.centers.shape}")
        return self._predictClusters(data)

    def fitPredict(self, data):
        """Compute kmeans centroids for trainData.
        
        Parameters:
            trainData: {np.ndarray, pd.DataFrame, list} of shape (n_samples, n_features)
                Training instances to compute cluster centers and to infer labels from.

        Returns:
            labels: np.ndarray of shape (n_samples,) containing int data with the cluster
                index for each sample in data
        """
        self.fit(data)
        return self._predictClusters(data)

    """
    Subfunctions
    """

    def _initializeCenters(self, data):
        """
        Initialize centers with method according to self.init
        """
        if self.init == 'random':
            randomRowIdxs = np.random.choice(data.shape[0], self.numberOfClusters)
            self.centers = data[randomRowIdxs]
        elif self.init == 'first':
            self.centers = data[:self.numberOfClusters]
        else:
            raise ValueError(f"Init parameter {self.init} not supported")
        if self.verbose:
            print("Initialization complete")

    def _convertToNumpy(self, data):
        """
        Converts data to numpy if the data is an object of pd.DataFrame or a list
        of lists. Otherwise, raise ValueError.
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, list):
            data = np.array(data)
            if len(data) == 2:
                return data
            else:
                raise ValueError(f"Expected a 2D list as input")
        raise ValueError(f"type {type(data)} not supported")

    @staticmethod
    def _computeNewCenter(trainData, clusterLabels, clusterIdx, currentCenter):
        """
        Computes new center as the mean vector of all the data points in a cluster
        as defined by:
        
        .. math::
            c_i = \frac{1}{\left | S_{i} \right |}\sum_{x_i\in S_i}x_i
        """
        return np.mean(trainData[clusterLabels == clusterIdx], axis=0)

    def _updateCenters(self, trainData, clusterLabels):
        """
        Iterator wrapper function to call _computeNewCenter for each center.
        """
        for clusterIdx, center in enumerate(self.centers):
            self.centers[clusterIdx] = self._computeNewCenter(trainData, clusterLabels, clusterIdx, center)

    def _predictClusters(self, data):
        """
        Compute the distances from each of the data points in data to each of the centers.
        data is of shape (n_samples, n_features) and centers is of shape (n_clusters, n_features).
        It will result in a l2distances matrix of shape (n_samples, n_clusters) of which the
        argmin function will return a (n_samples,) vector with the cluster assignation.
        """
        l2distances = cdist(data, self.centers)
        return np.argmin(l2distances, axis=1)

    def _stopIteration(self, previousCentroids, newCentroids, previousLabels, newLabels):
        """
        This function will return True if the centers have not changed or the data points 
        are in the same cluster than the previous iteration.
        """
        return self._centroidsNotChanged(previousCentroids, newCentroids) or self._pointsInSameCluster(previousLabels, newLabels)

    def _centroidsNotChanged(self, previousCentroids, newCentroids):
        """
        Computes distance between current and previous centroids and thresholds it with a tolerance
        """
        iterationDistance = np.sum(np.abs(newCentroids - previousCentroids))/previousCentroids.shape[1]
        if self.verbose:
            print(f"Centers have changed: {iterationDistance}")
        return iterationDistance < self.maxStopDistance

    def _pointsInSameCluster(self, previousLabels, newLabels):
        """
        Computes similarity and returns true if all previous labels have not been changed
        in this iteration.
        """
        boolArray = previousLabels == newLabels
        if self.verbose:
            print(f"Classifications changed: {np.sum(previousLabels != newLabels)}/{len(previousLabels)}")
        return np.all(boolArray)

    def _computeInertia(self, data, dataLabels):
        """
        Inertia computation. The inertia is defined as the sum of the distance from each data
        point to the closest center (the center of the cluster where they are assigned).

        .. math::
            I = \sum_{\forall point} d(point, cr)
        
        where cr is the closest center to the point.
        """
        inertia = 0.0
        for clusterIdx in range(self.numberOfClusters):
            clusterData = data[dataLabels == clusterIdx]
            clusterCenter = self.centers[clusterIdx]
            inertia += np.sum(l2dist(clusterData, clusterCenter))
        return inertia
