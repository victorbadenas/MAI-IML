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


def computeBounds(p, z, n):
    num = p + z**2/(2*n)
    den = 1 + z**2/n
    rad = z * np.sqrt((p*(1-p))/n + (z/(2*n)**2))
    upper = (num + rad) / den
    lower = (num - rad) / den
    return lower, upper


def ib3(X, labels, rand=True):

    if rand:
        X, labels = randomizeData(X, labels)

    CD = np.array([0])

    classificationAttempts = np.ones(1)
    classificationPositive = np.ones(1)
    z_acc = .9
    z_rej = .7

    for idx, (x, label) in enumerate(zip(X[1:], labels[1:]), 1):
        label = int(label)

        distance = cdist(x[None, :], np.atleast_2d(X[CD])).flatten()
        acc_low, _ = computeBounds(classificationPositive / classificationAttempts, z_acc, classificationAttempts)
        _, freq_high = computeBounds(classificationAttempts / np.sum(classificationAttempts), z_acc, classificationAttempts)
        acceptable = acc_low - freq_high >= 0

        if np.any(acceptable):
            y_min_idx = np.argmin(distance * acceptable)
        else:
            randomidx = np.random.randint(0, len(CD))
            sim_sort_idx = np.argsort(distance)
            y_min_idx = sim_sort_idx[-randomidx]

        if label != labels[CD][y_min_idx]:
            CD = np.append(CD, idx)
            classificationAttempts = np.append(classificationAttempts, np.array([0]))
            classificationPositive = np.append(classificationPositive, np.array([0]))
            distance = cdist(x[None, :], np.atleast_2d(X[CD])).flatten()

        att = (distance <= distance[-1]).astype(np.int)
        pos = att * (labels[CD] == labels[CD[-1]]).astype(np.int)
        classificationAttempts += att
        classificationPositive += pos

        if len(CD) > 1:
            acc_low, _ = computeBounds(classificationPositive / classificationAttempts, z_rej, classificationAttempts)
            _, freq_high = computeBounds(classificationAttempts / np.sum(classificationAttempts), z_rej, classificationAttempts)
            acceptable = acc_low - freq_high >= 0
            conserve_idx = np.logical_or(~att.astype(bool), acceptable)
            if np.any(conserve_idx):
                CD = CD[conserve_idx]
                classificationAttempts = classificationAttempts[conserve_idx]
                classificationPositive = classificationPositive[conserve_idx]

    acc_low, _ = computeBounds(classificationPositive / classificationAttempts, z_rej, classificationAttempts)
    _, freq_high = computeBounds(classificationAttempts / np.sum(classificationAttempts), z_rej, classificationAttempts)
    acceptable = acc_low - freq_high >= 0
    if np.any(acceptable):
        CD = CD[acceptable]

    return X[CD], labels[CD]
