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

    step1ResultsFolder = Path(config["resultsDir"]) / "phase2"
    step1ResultsFolder.mkdir(exist_ok=True, parents=True)
    data, trueLabels = loadArffFile(Path(config["path"]))
    data = data.to_numpy()

    pca = PCA(data.shape[1])
    reducedData = pca.fit_transform(data)
    print(len(pca.varianceRatios))
    print(f"Explained Ratio Variance {np.round(pca.varianceRatios, 2)}")
    #print(f"Explained Ratio Variance {np.round(pca.explained_variance_ratio_, 2)}")
    #print(f"Sum {np.sum(pca.explained_variance_ratio_)}")
    #print(f"Mean {pca.mean_}")
    #print(f"Variance {ipca.var_}")
    #0print(f"Noise Variance {pca.noise_variance_}")
    #Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"caca2_dims_pcaScatter.png")
