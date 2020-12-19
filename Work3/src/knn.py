import sys
import numpy as np
from sklearn_relief import ReliefF
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_classif
sys.path.append('src/')

from utils import convertToNumpy, ndcorrelate
import metrics
eps = np.finfo(float).eps

# distance metrics
COSINE = 'cosine'
MINKOWSKI = 'minkowski'
EUCLIDEAN = 'minkowski2'
DISTANCE_METRICS = [COSINE, MINKOWSKI, EUCLIDEAN]

# voting options
MAJORITY = 'majority'
INVERSE_DISTANCE_WEIGHTED = 'idw'
SHEPARDS_WORK = 'sheppards'
VOTING = [MAJORITY, INVERSE_DISTANCE_WEIGHTED, SHEPARDS_WORK]

# weights
UNIFORM = 'uniform'
RELIEFF = "relieff"
MUTUAL_INFO = 'mutual_info'
CORRELATION = "correlation"
WEIGHTS = [UNIFORM, MUTUAL_INFO, CORRELATION]

# distance computation methods
SCIPY = 'scipy'
MAT = 'mat'
DISTANCE_METHODS = [SCIPY, MAT]


class kNNAlgorithm:
    def __init__(self, n_neighbors=5,
                 *, weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 method='mat'):

        self.k = n_neighbors
        self.voting = voting
        self.weights = weights
        self.metric = metric
        self.p = p
        self.method = method
        self._validateParameters()

    def _computeFeatureWeights(self):
        if self.weights == UNIFORM:
            self.w = np.ones((self.trainX.shape[1],))
        elif self.weights == RELIEFF:
            self.w = ReliefF().fit(self.trainX, self.trainLabels).w_
        elif self.weights == MUTUAL_INFO:
            self.w = mutual_info_classif(self.trainX, self.trainLabels)
        elif self.weights == CORRELATION:
            self.w = ndcorrelate(self.trainX, self.trainLabels)
            self.w[self.w < 0] = 0
            if np.sum(self.w) == 0:
                print("Correlation weights sum 0, defaulting to uniform weights")
                self.w = np.ones((self.trainX.shape[1],))
        self.w = self.w / self.w.max()

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
        self._computeFeatureWeights()
        return self

    def _predict(self, X):
        X = convertToNumpy(X)
        distanceMatrix = self._computeDistanceMatrix(X)
        knnIndexes = self._computeKNNIndex(distanceMatrix)
        knnLabels = self._extractLabels(knnIndexes)
        decision = self._decide(knnLabels, distanceMatrix)
        return decision

    def _extractLabels(self, knnIndexes):
        labels = self.trainLabels[knnIndexes]
        return labels.astype(np.int)

    def _decide(self, knnLabels, distanceMatrix):
        if self.voting == MAJORITY:
            votingWeights = np.ones_like(knnLabels)
        elif self.voting == INVERSE_DISTANCE_WEIGHTED:
            votingWeights = 1 / (distanceMatrix[:, :self.k] + eps) ** self.p
        elif self.voting == SHEPARDS_WORK:
            votingWeights = np.exp(-1*distanceMatrix[:, :self.k])
        return self._computeDecision(knnLabels, votingWeights)

    def _computeDecision(self, knnLabels, votingWeights):
        numClasses = int(self.trainLabels.max()) + 1
        votes = np.empty((numClasses, *knnLabels.shape), dtype=int)
        for classNum in range(numClasses):
            votes[classNum] = np.where(knnLabels == classNum, 1, 0)
        weightedVotes = np.expand_dims(votingWeights, axis=0) * votes
        finalVotesPerClass = np.sum(weightedVotes, axis=2).T
        return np.argmax(finalVotesPerClass, axis=1)

    def _computeKNNIndex(self, distanceMatrix):
        knnIndex = [None]*distanceMatrix.shape[0]
        for i in range(distanceMatrix.shape[0]):
            knnIndex[i] = np.argsort(distanceMatrix[i, :])[:self.k]
        return np.vstack(knnIndex)

    def _computeDistanceMatrix(self, X):
        if self.method == MAT:
            return self._matricialDistanceMatrix(X)
        elif self.method == SCIPY:
            return self._scipyDistanceMatrix(X)

    def _scipyDistanceMatrix(self, X):
        if self.metric == EUCLIDEAN:
            return cdist(X, self.trainX, metric=MINKOWSKI, p=2, w=self.w)
        elif self.metric == MINKOWSKI:
            return cdist(X, self.trainX, metric=MINKOWSKI, p=1, w=self.w)
        return cdist(X, self.trainX, metric=self.metric, w=self.w)

    def _matricialDistanceMatrix(self, X):
        if self.metric == COSINE:
            return metrics.cosineDistance(X, self.trainX, w=self.w)
        elif self.metric == MINKOWSKI:
            return metrics.minkowskiDistance(X, self.trainX, w=self.w, p=1)
        elif self.metric == EUCLIDEAN:
            return metrics.euclideanDistance(X, self.trainX, w=self.w)

    def _validateParameters(self):
        assert self.k > 0, f"n_neighbors must be positive, not \'{self.k}\'"
        assert self.p > 0 and isinstance(self.p, int), f"p for distance voting must be a positive int"
        assert self.voting in VOTING, f"voting \'{self.voting}\' type not supported"
        assert self.weights in WEIGHTS, f"weights \'{self.weights}\' type not supported"
        assert self.metric in DISTANCE_METRICS, f"distance metric \'{self.metric}\' type not supported"
        assert self.method in DISTANCE_METHODS, f"distance computation method \'{self.method}\' not supported"

    def get_params(self, deep=True):
        return {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski', 'voting': 'majority', 'p': 1}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    N_features = 2
    classgs = np.array([
        ((0.75, 0.75) + tuple([0.75]*(N_features-2))),
        ((0.25, 0.25) + tuple([0.75]*(N_features-2))),
        ((0.75, 0.25) + tuple([0.75]*(N_features-2))),
        ((0.25, 0.75) + tuple([0.75]*(N_features-2)))
    ])

    data = []
    labels = []
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.75, 0.75) + tuple([0.75]*(N_features-2))))
    labels.append(np.zeros((50,)))
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.25, 0.25) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((50,), 1))
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.75, 0.25) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((50,), 2))
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.25, 0.75) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((50,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    newData = np.random.rand(50, N_features)
    newLabels = np.argmin(cdist(newData, classgs), axis=1)

    def plotModelTrial(trainData, testData, trainLabels, testLabels, classgs):
        plt.figure(figsize=(15, 9))
        for label, c in zip(np.unique(trainLabels), 'rgby'):
            subData = trainData[trainLabels == label]
            subNewData = testData[testLabels == label]
            plt.scatter(subData[:, 0], subData[:, 1], c=c, marker='+')
            plt.scatter(subNewData[:, 0], subNewData[:, 1], c=c, marker='x')
        # plt.scatter(classgs[:, 0], classgs[:, 1], c='k', marker='o')
        plt.vlines(0.5, 0, 1, colors='k', linestyles='dashed')
        plt.hlines(0.5, 0, 1, colors='k', linestyles='dashed')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([i/4 for i in range(5)])
        plt.yticks([i/4 for i in range(5)])
        plt.grid('on')

    plotModelTrial(data, newData, labels, newLabels, classgs)
    plt.show()

    print(f"train dataset size: {data.shape}, test dataset size: {newData.shape}")
    for d in DISTANCE_METRICS:
        for v in VOTING:
            for w in WEIGHTS:
                for m in [MAT]:
                    print(f"distance: {d}, voting: {v}, weights: {w}, method {m}")
                    knn = kNNAlgorithm(metric=d, voting=v, weights=w, method=m)
                    pred_labels = knn.fit(data, labels).predict(newData)
                    print(pred_labels)
