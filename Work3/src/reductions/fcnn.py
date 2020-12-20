import sys
import numpy as np
sys.path.append('..')

from knn import kNNAlgorithm

def fcnn(X, y):
    S, nearest, rep = [], {}, {}
    deltaS = __centroids(X, y)
    while len(deltaS) > 0:
        S.extend(deltaS)
        __restartRep(rep, S)
        for q in [q for q in range(len(X)) if q not in S]:
            __findNearestToQ(nearest, deltaS, q, X)
            __findMostRepresentativeToNearest(rep, nearest, y, q, X)
        deltaS = __recomputeDeltaS(rep, S)
    return X[S], y[S]

def __centroids(T, y):
    deltaS, nInstances = [], 0
    for c in np.unique(y):
        classInstances = T[np.where(y == c)]
        centroid = np.mean(classInstances, axis=0)
        distanceMatrix = kNNAlgorithm.computeDistanceMatrix(classInstances, centroid[np.newaxis, :], w=np.ones((len(centroid),)))
        closest2Centroid = np.argsort(distanceMatrix, axis=None)[0]
        deltaS.append(np.where(T == classInstances[closest2Centroid])[0][0])
        nInstances += len(classInstances)
    return deltaS

def __restartRep(rep, S):
    for p in S:
        if p in rep:
            del rep[p]

def __findNearestToQ(nearest, deltaS, q, X):
    weights = np.ones((len(X[0]),))
    qX = X[q][np.newaxis, :]
    distanceMatrix = kNNAlgorithm.computeDistanceMatrix(X[deltaS], qX, w=weights)
    closest = np.argsort(distanceMatrix, axis=None)[0]
    if q not in nearest or (q in nearest and kNNAlgorithm.computeDistanceMatrix(X[nearest[q]][np.newaxis, :], qX, w=weights)[0] > distanceMatrix[closest]):
        nearest[q] = deltaS[closest]

def __findMostRepresentativeToNearest(rep, nearest, y, q, X):
    weights = np.ones((len(X[0]),))
    nearestX = X[nearest[q]][np.newaxis, :]
    closest = kNNAlgorithm.computeDistanceMatrix(nearestX, X[q][np.newaxis, :], w=weights)[0]
    if y[q] != y[nearest[q]] and \
        (nearest[q] not in rep or \
         (nearest[q] in rep and kNNAlgorithm.computeDistanceMatrix(nearestX, X[rep[nearest[q]]][np.newaxis, :], w=weights)[0] > closest)):
        rep[nearest[q]] = q

def __recomputeDeltaS(rep, S):
    deltaS = []
    for p in S:
        if p in rep and rep[p] not in deltaS:
            deltaS.append(rep[p])
    return deltaS
