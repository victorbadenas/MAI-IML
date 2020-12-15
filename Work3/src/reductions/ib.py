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
        if label != labels[closer_y_idx]:
            CD.append(idx)
    return X[CD], labels[CD]


def compute(p, z, n):
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
        acc_low, _ = compute(classificationPositive / classificationAttempts, z_acc, classificationAttempts)
        _, freq_high = compute(classificationAttempts / np.sum(classificationAttempts), z_acc, classificationAttempts)
        acceptable = acc_low - freq_high >= 0

        if np.any(acceptable):
            y_min_idx = np.argmin(distance * acceptable)
        else:
            randomidx = np.random.randint(0, len(CD))
            sim_sort_idx = np.argsort(distance)
            y_min_idx = sim_sort_idx[-randomidx]

        if label != labels[y_min_idx]:
            CD = np.append(CD, idx)
            classificationAttempts = np.append(classificationAttempts, np.array([0]))
            classificationPositive = np.append(classificationPositive, np.array([0]))
            distance = cdist(x[None, :], np.atleast_2d(X[CD])).flatten()

        att = (distance <= distance[-1]).astype(np.int)
        pos = att * (labels[CD] == labels[CD[-1]]).astype(np.int)
        classificationAttempts += att
        classificationPositive += pos

        if len(CD) > 1:
            acc_low, _ = compute(classificationPositive / classificationAttempts, z_rej, classificationAttempts)
            _, freq_high = compute(classificationAttempts / np.sum(classificationAttempts), z_rej, classificationAttempts)
            acceptable = acc_low - freq_high >= 0
            conserve_idx = np.logical_or(~att.astype(bool), acceptable)
            if np.any(conserve_idx):
                CD = CD[conserve_idx]
                classificationAttempts = classificationAttempts[conserve_idx]
                classificationPositive = classificationPositive[conserve_idx]

    acc_low, _ = compute(classificationPositive / classificationAttempts, z_rej, classificationAttempts)
    _, freq_high = compute(classificationAttempts / np.sum(classificationAttempts), z_rej, classificationAttempts)
    acceptable = acc_low - freq_high >= 0
    if np.any(acceptable):
        CD = CD[acceptable]

    return X[CD], labels[CD]

    # for x, label in zip(X[1:], labels[1:]):
    #     sim = -1*cdist(x[None, :], CD).flatten()
    #     acceptable = sim >= th
    #     if np.any(acceptable):
    #         ymax_idx = np.random.choice(np.where(acceptable)[0])
    #         print('acceptable')
    #     else:
    #         idx = np.random.randint(CD.shape[0])
    #         sim_sort_idx = np.argsort(sim)
    #         ymax_idx = sim_sort_idx[-idx]
    #         print('random')
    #     if label == cdlabels[ymax_idx]:
    #         correct = True
    #         record[ymax_idx] += 1
    #     else:
    #         correct = False
    #         CD = np.concatenate([CD, x[None, :]])
    #         cdlabels = np.concatenate([cdlabels, np.array(label).reshape(-1)])

    #     sim2 = -1*cdist(x[None, :], CD).flatten()
    #     y_indexes = np.where(sim2 >= sim[ymax_idx])[0]
    #     y_to_remove = y_indexes[np.where(record[y_indexes] < 2)[0]]
    #     CD = np.delete(CD, y_to_remove, axis=0)
    #     cdlabels = np.delete(cdlabels, y_to_remove)
    return CD, cdlabels
