import sys
sys.path.append('src/')

from knn import kNNAlgorithm, DISTANCE_METRICS, VOTING, WEIGHTS
from utils import convertToNumpy
from reductions import ib2, enn, fcnn

IB2 = 'ib2'
ENN = 'enn'
FCNN = 'fcnn'
NONE = None
REDUCTION_METHODS = [IB2, ENN, FCNN, NONE]


class reductionKnnAlgorithm(kNNAlgorithm):
    def __init__(self, n_neighbors=5,
                 *, weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 reduction=NONE):

        super(reductionKnnAlgorithm, self).__init__(n_neighbors=n_neighbors,
                                                    weights=weights,
                                                    metric=metric,
                                                    voting=voting,
                                                    p=p)
        assert reduction in REDUCTION_METHODS, f"reduction method {reduction} not supported"
        self.reduction = reduction

    def _fit(self, X, y):
        assert X.shape[0] >= self.k, f"Need a minimum of {self.k} points"
        X, y = convertToNumpy(X.copy()), convertToNumpy(y.copy())
        self.trainX, self.trainLabels = self._reduceInstances(X, y)
        self._computeFeatureWeights()
        return self

    def _reduceInstances(self, X, y):
        if self.reduction == IB2:
            return ib2(X, y)
        elif self.reduction == ENN:
            return enn(X, y, n_neighbors=self.k, metric=self.metric)
        elif self.reduction == FCNN:
            return fcnn(X, y)
        elif self.reduction == NONE:
            return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist
    data = []
    labels = []
    data.append(np.random.rand(50, 2)/2 - 0.25 + (0.75, 0.75))
    labels.append(np.zeros((50,)))
    data.append(np.random.rand(50, 2)/2 - 0.25 + (0.25, 0.25))
    labels.append(np.full((50,), 1))
    data.append(np.random.rand(50, 2)/2 - 0.25 + (0.75, 0.25))
    labels.append(np.full((50,), 2))
    data.append(np.random.rand(50, 2)/2 - 0.25 + (0.25, 0.75))
    labels.append(np.full((50,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    classgs = np.array([[0.75, 0.75], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])
    newData = np.random.rand(10, 2)
    newLabels = np.argmin(cdist(newData, classgs), axis=1)

    def plotModelTrial(trainData, testData, trainLabels, testLabels, classgs):
        plt.figure(figsize=(15, 9))
        for label, c in zip(np.unique(trainLabels), 'rgby'):
            subData = trainData[trainLabels == label]
            subNewData = testData[testLabels == label]
            plt.scatter(subData[:, 0], subData[:, 1], c=c, marker='+')
            plt.scatter(subNewData[:, 0], subNewData[:, 1], c=c, marker='x')
        plt.vlines(0.5, 0, 1, colors='k', linestyles='dashed')
        plt.hlines(0.5, 0, 1, colors='k', linestyles='dashed')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([i/4 for i in range(5)])
        plt.yticks([i/4 for i in range(5)])
        plt.grid('on')

    for r in REDUCTION_METHODS:
        print(f"reduction: {r}")
        rknn = reductionKnnAlgorithm(reduction=r)
        pred_labels = rknn.fit(data, labels).predict(newData)
        accuracy = np.average(newLabels == pred_labels)
        print(pred_labels)
        print(f"Accuracy: {accuracy}")
        plotModelTrial(data, newData, labels, newLabels, classgs)
        for label, c in zip(np.unique(labels), 'rgby'):
            ibData = rknn.trainX[rknn.trainLabels == label]
            plt.scatter(ibData[:, 0], ibData[:, 1], c=c, marker='o')
        plt.show()
