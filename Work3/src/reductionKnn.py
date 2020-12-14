import sys
sys.path.append('src/')

from knn import kNNAlgorithm, DISTANCE_METRICS, VOTING, WEIGHTS
from utils import convertToNumpy
from reductions import ib2, ib3

IB2 = 'ib2'
IB3 = 'ib3'
NONE = None
REDUCTION_METHODS = [IB2, IB3, NONE]


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
        if self.reduction == IB3:
            return ib3(X, y)
        elif self.reduction == NONE:
            return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist
    data = []
    labels = []
    data.append(np.random.rand(50, 3) - 0.5 + (1, 1, 1))
    labels.append(np.zeros((50,)))
    data.append(np.random.rand(50, 3) - 0.5 + (0, 0, 0))
    labels.append(np.full((50,), 1))
    data.append(np.random.rand(50, 3) - 0.5 + (1, 0, 1))
    labels.append(np.full((50,), 2))
    data.append(np.random.rand(50, 3) - 0.5 + (0, 1, 1))
    labels.append(np.full((50,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    classgs = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 1]])
    newData = 1.5*np.random.rand(10, 3)
    newLabels = np.argmin(cdist(newData, classgs), axis=1)

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection='3d')
    for label, c in zip(np.unique(labels), 'rgby'):
        subData = data[labels == label]
        subNewData = newData[newLabels == label]
        ax.scatter(subData[:, 0], subData[:, 1], subData[:, 2], c=c)
        ax.scatter(subNewData[:, 0], subNewData[:, 1], subNewData[:, 2], c=c, marker='x')
    ax.scatter(classgs[:, 0], classgs[:, 1], classgs[:, 2], c='k', marker='o')
    plt.show()

    for d in DISTANCE_METRICS:
        for v in VOTING:
            for w in WEIGHTS:
                for r in REDUCTION_METHODS:
                    print(f"distance: {d}, voting: {v}, weights: {w}, reduction: {r}")
                    rknn = reductionKnnAlgorithm(metric=d, voting=v, weights=w, reduction=r)
                    pred_labels = rknn.fit(data, labels).predict(newData)
                    accuracy = np.average(newLabels == pred_labels)
                    print(pred_labels)
                    print(f"Accuracy: {accuracy}")

