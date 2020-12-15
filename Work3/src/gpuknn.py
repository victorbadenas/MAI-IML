import sys
import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf
sys.path.append('src/')
tf.autograph.set_verbosity(0)
gpu_devices = tf.config.experimental.list_logical_devices('GPU')
print("Running with {}".format(gpu_devices[0] if len(gpu_devices) > 0 else 'cpu'))

from knn import *


class gpukNNAlgorithm(kNNAlgorithm):
    def _computeDistanceMatrix(self, X):
        if len(gpu_devices) > 0:
            return self._gpuDistanceMatrix(X)
        return self._cpuDistanceMatrix(X)

    def _cpuDistanceMatrix(self, X):
        if self.metric == EUCLIDEAN:
            return cdist(X, self.trainX, metric=MINKOWSKI, p=2, w=self.w)
        elif self.metric == MINKOWSKI:
            return cdist(X, self.trainX, metric=MINKOWSKI, p=1, w=self.w)
        return cdist(X, self.trainX, metric=self.metric, w=self.w)

    def _gpuDistanceMatrix(self, X):
        with tf.device('/GPU:0'):
            trainTensor = tf.constant(self.trainX)
            Xtensor = tf.constant(X)
            weightsTensor = tf.constant(self.w)
            if self.metric == COSINE:
                trainTensor = trainTensor * tf.sqrt(weightsTensor[tf.newaxis, :])
                Xtensor = Xtensor * tf.sqrt(weightsTensor[tf.newaxis, :])
                tfd = tf.matmul(trainTensor, tf.transpose(Xtensor))
                tfd /= tf.norm(trainTensor, axis=1, keepdims=True)
                tfd /= tf.norm(Xtensor, axis=1)[tf.newaxis, :]
                tfd = 1 - tfd
            else:
                Xtensor = tf.repeat(tf.expand_dims(Xtensor, 0), trainTensor.shape[0], axis=0)
                trainTensor = tf.repeat(tf.expand_dims(trainTensor, 1), Xtensor.shape[1], axis=1)
                weightsTensor = tf.repeat(tf.expand_dims(weightsTensor, 0), Xtensor.shape[1], axis=0)
                weightsTensor = tf.repeat(tf.expand_dims(weightsTensor, 0), Xtensor.shape[0], axis=0)
                tfd = trainTensor - Xtensor
                if self.metric == MINKOWSKI:
                    tfd = tf.math.pow(tf.math.reduce_sum(weightsTensor * tf.math.pow(tf.abs(tfd), self.p), axis=2), (1/self.p))
                elif self.metric == EUCLIDEAN:
                    tfd = tf.math.pow(tf.math.reduce_sum(weightsTensor * tf.math.pow(tfd, 2), axis=2), .5)
            return tfd.numpy().T

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist
    data = []
    labels = []
    data.append(np.random.rand(500, 2)/2 - 0.25 + (0.75, 0.75))
    labels.append(np.zeros((500,)))
    data.append(np.random.rand(500, 2)/2 - 0.25 + (0.25, 0.25))
    labels.append(np.full((500,), 1))
    data.append(np.random.rand(500, 2)/2 - 0.25 + (0.75, 0.25))
    labels.append(np.full((500,), 2))
    data.append(np.random.rand(500, 2)/2 - 0.25 + (0.25, 0.75))
    labels.append(np.full((500,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    classgs = np.array([[0.75, 0.75], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])
    newData = np.random.rand(100, 2)
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

    for d in DISTANCE_METRICS:
        for v in VOTING:
            for w in WEIGHTS:
                print(f"now evaluating: distance: {d}, voting: {v}, weights: {w}")
                knn = gpukNNAlgorithm(metric=d, voting=v, weights=w)
                pred_labels = knn.fit(data, labels).predict(newData)
                accuracy = np.average(newLabels == pred_labels)
                print(f"predicted labels: {pred_labels}")
                print(f"Accuracy: {accuracy}")
