import numpy as np
from sklearn.metrics import cluster as clusteringMetrics

def purity_score(y_true, y_pred):
    contingency_matrix = clusteringMetrics.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
