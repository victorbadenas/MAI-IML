import sys
sys.path.append(".")

import argparse
import json
import numpy as np
from pathlib import Path
from src.dataset import ArffFile
from src.visualize import Visualizer
from src.pca import PCA
#from sklearn.decomposition import IncrementalPCA as PCA
#from sklearn.decomposition import PCA


def loadArffFile(arffFilePath):
    # load arff files
    arffFile = ArffFile(arffFilePath)
    unsupervisedFeatures = arffFile.getData().copy()

    # remove label column from training data and separate it
    labelColumn = unsupervisedFeatures.columns[-1]
    unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
    y = arffFile.getData()[labelColumn]

    # return features and labels
    return unsupervisedFeatures, y

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--jsonConfig", type=Path, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()
    with open(args.jsonConfig, 'r') as f:
        config = json.load(f)

    data, trueLabels = loadArffFile(Path(config["path"]))
    X = data.to_numpy()
    mean_ = np.mean(X, axis=0)
    X -= mean_

    # Compute covariance matrix
    C = np.dot(X.T, X) / (len(X)-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # SVD
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    # Relationship between singular values and eigen values:
    #print(np.allclose(np.square(Sigma) / (n - 1), eigen_vals)) # True

    idxs = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[idxs]
    eigen_vecs = eigen_vecs[:, idxs]

    #eigen_vecs = eigen_vecs[:, :3]
    #eigen_vals = eigen_vals[:3]

    print(np.dot(X, eigen_vecs[:, :3]))
    print(np.dot(U[:, :3], np.diag(Sigma[:3])))
    
    variance = (Sigma ** 2) / (len(X) - 1)
    explained_variance_ratio_ = variance / np.sum(variance)
    print(explained_variance_ratio_)
    print(explained_variance_ratio_[:3])
    
    explained_variance_ratio_ = eigen_vals / np.sum(eigen_vals)
    print(explained_variance_ratio_)
    print(explained_variance_ratio_[:3])