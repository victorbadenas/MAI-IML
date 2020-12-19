import numpy as np
from scipy.spatial.distance import cdist
np.random.seed(0)


def randomizeData(X, labels):
    randidx = np.random.permutation(np.arange(len(X)))
    X = X[randidx]
    labels = labels[randidx]
    return X, labels


def ib2(X, labels, rand=True):
    if rand:
        X, labels = randomizeData(X, labels)

    CD = [0]
    for idx, (x, label) in enumerate(zip(X[1:], labels[1:]), 1):
        distance = cdist(x[None, :], np.atleast_2d(X[CD]))
        closer_y_idx = np.argmin(distance)
        if label != labels[CD][closer_y_idx]:
            CD.append(idx)
    return X[CD], labels[CD]
