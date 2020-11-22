import numpy as np
from scipy.spatial.distance import cdist
from utils import convertToNumpy


# distance metrics
COSINE = 'cosine'
MINKOWSKI = 'minkowski'
DISTANCE_METRICS = [COSINE, MINKOWSKI]

# voting options
MAJORITY = 'majority'
INVERSE_DISTANCE_WEIGHTED = 'idw'
SHEPARDS_WORK = 'sheppards'
VOTING = [MAJORITY, INVERSE_DISTANCE_WEIGHTED, SHEPARDS_WORK]

# weights
UNIFORM = 'uniform'
DISTANCE = 'distance'
WEIGHTS = [UNIFORM, DISTANCE]


class KNN:
    def __init__(self, n_neighbors=5,
                 *, weights='uniform',
                 metric='minkowski',
                 mink_r=1, voting=''):

        self.validateParameters(n_neighbors, voting, weights, metric)
        self.k = n_neighbors
        self.voting = voting
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def _fit(self, X, y):
        assert X.shape[0] >= self.k, f"Need a minimum of {self.k} points"
        self.trainX = convertToNumpy(X.copy())
        self.trainLabels = convertToNumpy(y.copy())
        return self

    def _predict(self, X):
        raise NotImplementedError

    def computeDistanceMatrix(self, X):
        return cdist(X, self.trainX, metric=self.metric)

    def validateParameters(self, k, voting, weigths, metric):
        assert k > 0, f"n_neighbors must be positive, not \'{k}\'"
        assert voting in VOTING, f"voting \'{voting}\'type not supported"
        assert weigths in WEIGHTS, f"weights \'{weigths}\'type not supported"
        assert metric in DISTANCE_METRICS, f"distance metric \'{metric}\'type not supported"


if __name__ == "__main__":
    data = np.array()
    knn = KNN()
    knn.fit(data)
