import copy
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from ..utils import convertToNumpy


class FCM:
    """Fuzzy CMeans Clustering algorithm:

    This object is responsible of performing the fcm algorithm in a
    set of data and compute its centers.

    Parameters:
        n_clusters : int, default=8
            The number of clusters to form as well as the number of
            centroids to generate.

        max_iter : int, default=300
            Maximum number of iterations of the fcm algorithm for a
            single run.

        tol : float, default=1e-4
            Maximum value tolerated to declare convergence by stability of the U matrix

        verbose : bool, default=False
            Verbosity mode.

    """
    def __init__(self, n_clusters=8, *, m=3, max_iter=500, tol=1e-4, verbose=False):
        self.nClusters = int(n_clusters)
        self.m = int(m)
        self.maxIter = max_iter
        self.tolerance = tol
        self.verbose = verbose
        assert self.m > 1
        assert self.nClusters > 0
        self.reset()

    def reset(self):
        self.currentU = None
        self.centers = None
        self.labels = None

    def fit(self, trainData):
        """Compute cmeans centroids for trainData.
        
        Parameters:
            trainData: {np.ndarray, pd.DataFrame, list} of shape (n_samples, n_features)
                Training instances to compute cluster centers

        Returns:
            self: fitted algorithm
        """
        trainData = convertToNumpy(trainData)
        self.currentU = self._initUMatrix(trainData)

        for _ in range(self.maxIter):
            previousU = copy.copy(self.currentU)
            self.centers = self._updateCenters(trainData)
            self.currentU = self._computeUMatrix(trainData)
            if self._distanceInTolerance(self.currentU, previousU):
                break

        self.labels = self.getLabels(self.currentU)
        return self

    def predict(self, data):
        """Compute cmeans labels for trainData given previously computed centroids.
        
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
        predictU = self._computeUMatrix(data)
        return self.getLabels(predictU)

    def fitPredict(self, data):
        """Compute cmeans centroids for trainData.
        
        Parameters:
            trainData: {np.ndarray, pd.DataFrame, list} of shape (n_samples, n_features)
                Training instances to compute cluster centers and to infer labels from.

        Returns:
            labels: np.ndarray of shape (n_samples,) containing int data with the cluster
                index for each sample in data
        """
        self.fit(data)
        return self.labels

    @staticmethod
    def getLabels(Umatrix):
        if Umatrix is None:
            raise ValueError("U Matrix not set, please run fcm.FCM().fit(X) first")
        return np.argmax(Umatrix, axis=-1)

    def _computeUMatrix(self, data):
        """
        compute Umatrix update as defined in page 30 of:
        http://openaccess.uoc.edu/webapps/o2/bitstream/10609/59066/7/ruizjcTFG0117memoria.pdf

        .. math:: 
            u_{ij} = \\frac {1}{\sum_{k=1}^{C} (\\frac{d_{ij}}{d_{ik}})^{\\frac{2}{m-1}}}
        """
        dij = cdist(data, self.centers) # distance of all data to all centers : shape (nSamples, nCenters)
        dik = np.repeat(dij[:, np.newaxis, :], self.nClusters, axis=1) # repeat for all C clusters
        denRatio = dij[:, :, np.newaxis] / dik # division through all axis
        denRatio = denRatio  ** (2/(self.m-1)) # power
        return 1 / np.sum(denRatio, axis=2) # sum though all clusters and inverse

    def _updateCenters(self, trainData):
        """
        compute centers as defined in:

        .. math:: 
            v_i = \\frac {\sum_{k=0}^{n-1}(u_{ik})^{m}x_i}{\sum_{k=0}^{n-1}(u_{ik})^{m}}
        
        the equation is vectorixed by multiplying U**m·X and then dividing by the sum 
        of each row of U**m.
        """
        uToM = self.currentU ** self.m
        den = np.sum(uToM.T, axis=1, keepdims=True)
        return np.dot(uToM.T, trainData) / den

    def _distanceInTolerance(self, currentU, previousU):
        """
        compute norm of the difference and threshold it by tolerance
        """
        return norm(currentU - previousU) < self.tolerance

    def _initUMatrix(self, trainData):
        """
        returns random array of shape (nSamples, nClusters) which, bu definition,
        the sum of all membershp values for a sample to each cluster must equal 1.
        For this goal, it is normalized by the sum of the random values obtained in
        each row of the matrix
        """
        U = np.random.rand(trainData.shape[0], self.nClusters)
        return U / np.sum(U, axis=1, keepdims=True)
