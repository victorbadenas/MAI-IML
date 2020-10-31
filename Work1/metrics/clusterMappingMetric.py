import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

def clusteringMappingMetric(labels_pred, labels_gs):
    """
    docstring
    """
    assert len(labels_gs) == len(labels_pred)

    labels_pred = np.array(labels_pred)
    labels_gs = np.array(labels_gs)
    modified_labels = labels_pred.copy()
    for clusterIdx in set(labels_pred):
        mask = labels_pred == clusterIdx
        clusterDataGs = labels_gs[mask]
        clusterAssignation = Counter(clusterDataGs).most_common(1)[0][0]
        modified_labels[mask] = clusterAssignation
    return confusion_matrix(labels_gs, modified_labels)
