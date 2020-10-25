import copy
import numpy as np
from scipy.spatial.distance import cdist
from ..utils import convertToNumpy

"""
http://openaccess.uoc.edu/webapps/o2/bitstream/10609/59066/7/ruizjcTFG0117memoria.pdf
Page 29

- Choose a number of clusters.
- Assign coefficients randomly to each data point for being in the clusters.
- Repeat until the algorithm has converged (that is, the coefficients' change between two iterations is no more than {\displaystyle \varepsilon }\varepsilon , the given sensitivity threshold) :
    - Compute the centroid for each cluster (shown below).
    - For each data point, compute its coefficients of being in the clusters.

"""

class FCM:
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
        self.currentU, self.centers = None, None

    def fit(self, trainData, returnLabels=False):
        # convert to numpy array
        trainData = convertToNumpy(trainData)

        # assign coefficients randomly to each data point for being in the clusters
        self.currentU = self._initUMatrix(trainData)

        for iterationIdx in range(self.maxIter):
            previousU = copy.copy(self.currentU)
            # compute the centroid for each cluster
            self.centers = self._updateCenters(trainData)
            # compute U matrix by predicting
            currentU = self._computeUMatrix(trainData)
            if self._distanceInTolerance(self.currentU, previousU):
                break
        if returnLabels:
            return np.argmax(self.currentU, axis=-1)

    def predict(self, data):
        raise NotImplementedError

    def fitPredict(self, data):
        return self.fit(data, returnLabels=True)

    def _computeUMatrix(self, data):
        raise NotImplementedError

    def _updateCenters(self, trainData):
        """
        compute centers as defined in:
        .. math:: 
        
            v_i = \frac {\sum_{k=0}^{n-1}(u_{ik})^{m}x_i}{\sum_{k=0}^{n-1}(u_{ik})^{m}}
        
        the equation is vectorixed by multiplying U**m·X and then dividing by the sum 
        of each row of U**m.
        """
        uToM = self.currentU ** self.m
        den = np.sum(uToM.T, axis=1, keepdims=True)
        return np.dot(uToM.T, trainData) / den

    def _distanceInTolerance(self, currentU, previousU):
        """
        compute distance and threshold it by tolerance
        """
        return cdist(currentU, previousU) < self.tolerance

    def _initUMatrix(self, trainData):
        """
        returns random array of shape (nSamples, nClusters) which, bu definition,
        the sum of all membershp values for a sample to each cluster must equal 1.
        For this goal, it is normalized by the sum of the random values obtained in
        each row of the matrix
        """
        U = np.random.rand(trainData.shape[0], self.nClusters)
        return U / np.sum(U, axis=1, keepdims=True)
