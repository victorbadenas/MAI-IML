import sys
sys.path.append('src/')

from knn import kNNAlgorithm, DISTANCE_METRICS, VOTING, WEIGHTS
from utils import convertToNumpy
from reductions import ib2reduction

IB2 = 'ib2'
NONE = None
REDUCTION_METHODS = [IB2, NONE]


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
            return ib2reduction(X, y)
        elif self.reduction == NONE:
            return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    data = []
    labels = []
    data.append(np.random.rand(50, 3) + (1, 1, 1))
    labels.append(np.zeros((50,)))
    data.append(np.random.rand(50, 3) + (0, 0, 0))
    labels.append(np.full((50,), 1))
    data.append(np.random.rand(50, 3) + (1, 0, 1))
    labels.append(np.full((50,), 2))
    data.append(np.random.rand(50, 3) + (0, 1, 1))
    labels.append(np.full((50,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    newData = 2*np.random.rand(10, 3)
    plt.figure(figsize=(15, 9))
    for label in np.unique(labels):
        subData = data[labels == label]
        plt.scatter(subData[:, 0], subData[:, 1])
    plt.scatter(newData[:, 0], newData[:, 1], c='k', marker='x')
    plt.show()

    for d in DISTANCE_METRICS:
        for v in VOTING:
            for w in WEIGHTS:
                for r in REDUCTION_METHODS:
                    print(f"distance: {d}, voting: {v}, weights: {w}, reduction: {r}")
                    rknn = reductionKnnAlgorithm(metric=d, voting=v, weights=w, reduction=r)
                    pred_labels = rknn.fit(data, labels).predict(newData)
                    print(pred_labels)
