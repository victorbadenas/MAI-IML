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

    CD = X[0].reshape(1, -1)
    cdlabels = np.array(labels[0]).reshape(-1)
    for x, label in zip(X[1:], labels[1:]):
        sim = cdist(x[None, :], CD)
        ymaxidx = np.argmax(sim)
        if label != cdlabels[ymaxidx]:
            CD = np.concatenate([CD, x[None, :]])
            cdlabels = np.concatenate([cdlabels, np.array(label).reshape(-1)])
    return CD, cdlabels


def ib3(X, labels, rand=True):
    if rand:
        X, labels = randomizeData(X, labels)

    CD = X[0].reshape(1, -1)
    cdlabels = np.array(labels[0]).reshape(-1)

    for x, label in zip(X[1:], labels[1:]):
        sim = cdist(x[None, :], CD)

    return X, labels
