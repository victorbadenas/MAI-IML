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
                 mink_r=1, voting='majority'):

        self.__validateParameters(n_neighbors, voting, weights, metric)
        self.k = n_neighbors
        self.voting = voting
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        return self.__fit(X, y)

    def predict(self, X):
        return self.__predict(X)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def __fit(self, X, y):
        assert X.shape[0] >= self.k, f"Need a minimum of {self.k} points"
        self.trainX = convertToNumpy(X.copy())
        self.trainLabels = convertToNumpy(y.copy())
        return self

    def __predict(self, X):
        distanceMatrix = self.__computeDistanceMatrix(X)
        raise NotImplementedError

    def __computeDistanceMatrix(self, X):
        return cdist(X, self.trainX, metric=self.metric)

    def __validateParameters(self, k, voting, weigths, metric):
        assert k > 0, f"n_neighbors must be positive, not \'{k}\'"
        assert voting in VOTING, f"voting \'{voting}\' type not supported"
        assert weigths in WEIGHTS, f"weights \'{weigths}\' type not supported"
        assert metric in DISTANCE_METRICS, f"distance metric \'{metric}\' type not supported"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = []
    labels = []
    data.append(np.random.rand(50, 2) + (1, 1))
    labels.append(np.zeros((50,)))
    data.append(np.random.rand(50, 2) + (0, 0))
    labels.append(np.full((50,), 1))
    data.append(np.random.rand(50, 2) + (1, 0))
    labels.append(np.full((50,), 2))
    data.append(np.random.rand(50, 2) + (0, 1))
    labels.append(np.full((50,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)
    try:
        knn = KNN()
        knn.fit(data, labels)
    except Exception as e:
        print(e)
    plt.figure(figsize=(15, 9))
    for label in np.unique(labels):
        subData = data[labels == label]
        plt.scatter(subData[:,0], subData[:,1])
    plt.show()
