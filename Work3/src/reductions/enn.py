import sys
import numpy as np
sys.path.append('..')

from knn import kNNAlgorithm

def enn(X, y, **knnparams):
    # Fit for X, y
    if 'n_neighbors' in knnparams:
        knnparams['n_neighbors'] += 1

    nn = kNNAlgorithm(**knnparams).fit(X, y)
    nnIndexes = nn._computeKNNIndex(nn.computeDistanceMatrix(X, nn.trainX, nn.w, nn.metric, nn.method))
    nnIndexes = nnIndexes[:, 1:]
    nnlabels = y[nnIndexes]
    yPred = nn._computeDecision(nnlabels, np.ones_like(nnlabels))

    under_mask = yPred == y
    return X[under_mask], y[under_mask]
