import sys
import numpy as np
from scipy.spatial.distance import cdist
sys.path.append('src/')
import logging

from knn import *
from utils import timer

DTYPES = [np.float, np.float16, np.float32, np.float64, np.double]

class matkNNAlgorithm(kNNAlgorithm):
    dtype = np.float16

    def _computeDistanceMatrix(self, X):
        return self._matricialDistanceMatrix(X)

    def _cpuDistanceMatrix(self, X):
        if self.metric == EUCLIDEAN:
            return cdist(X, self.trainX, metric=MINKOWSKI, p=2, w=self.w)
        elif self.metric == MINKOWSKI:
            return cdist(X, self.trainX, metric=MINKOWSKI, p=1, w=self.w)
        return cdist(X, self.trainX, metric=self.metric, w=self.w)

    def _matricialDistanceMatrix(self, X):
        X = X.astype(self.dtype)
        self.w = self.w.astype(self.dtype)
        self.trainX = self.trainX.astype(self.dtype)

        if self.metric == COSINE:
            trainX = self.trainX * np.sqrt(self.w)[None, :]
            X = X.copy() * np.sqrt(self.w)[None, :]
            d = X@trainX.T / np.linalg.norm(trainX, axis=1)[None, :] / np.linalg.norm(X, axis=1, keepdims=True)
            d = 1-d
        else:
            X = np.repeat(X[None, :], self.trainX.shape[0], axis=0)
            trainX = np.repeat(np.expand_dims(self.trainX.copy(), 1), X.shape[1], axis=1)
            weights = np.repeat(np.expand_dims(self.w.copy(), 0), X.shape[1], axis=0)
            weights = np.repeat(np.expand_dims(weights, 0), X.shape[0], axis=0)
            d = trainX - X
            if self.metric == MINKOWSKI:
                d = np.sum((weights * np.abs(d)**self.p), axis=2)**(1/self.p)
            elif self.metric == EUCLIDEAN:
                d = np.sqrt(np.sum(weights * d ** 2, axis=2))
            d = d.T
        return d

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    from pathlib import Path

    N_features = 14
    classgs = np.array([
        ((0.75, 0.75) + tuple([0.75]*(N_features-2))),
        ((0.25, 0.25) + tuple([0.75]*(N_features-2))),
        ((0.75, 0.25) + tuple([0.75]*(N_features-2))),
        ((0.25, 0.75) + tuple([0.75]*(N_features-2)))
    ])

    data = []
    labels = []
    data.append(np.random.rand(10000, N_features)/2 - 0.25 + ((0.75, 0.75) + tuple([0.75]*(N_features-2))))
    labels.append(np.zeros((10000,)))
    data.append(np.random.rand(10000, N_features)/2 - 0.25 + ((0.25, 0.25) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((10000,), 1))
    data.append(np.random.rand(10000, N_features)/2 - 0.25 + ((0.75, 0.25) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((10000,), 2))
    data.append(np.random.rand(10000, N_features)/2 - 0.25 + ((0.25, 0.75) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((10000,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    newData = np.random.rand(10000, N_features)
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

    # plotModelTrial(data, newData, labels, newLabels, classgs)
    # plt.show()

    print(f"train dataset size: {data.shape}, test dataset size: {newData.shape}")
    for d in [DISTANCE_METRICS[1]]:
        for v in VOTING:
            for w in WEIGHTS:
                print(f"now evaluating: distance: {d}, voting: {v}, weights: {w}")
                knn = matkNNAlgorithm(metric=d, voting=v, weights=w)
                pred_labels = knn.fit(data, labels).predict(newData)
                accuracy = np.average(newLabels == pred_labels)
                print(pred_labels)
                print(f"Accuracy: {accuracy}")
