import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore')
from src.utils import timer
from src.dataset import ArffFile
from src.visualize import Visualizer
from src.pca import PCA


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--jsonConfig", type=Path, required=True)
    return parser.parse_args()

class Main:
    def __init__(self, args):
        self.args = args
        with open(self.args.jsonConfig , 'r') as f:
            self.config = json.load(f)

    def __call__(self):
        self.step1()

    @staticmethod
    def loadArffFile(arffFilePath):
        # load arff files
        arffFile = ArffFile(arffFilePath)
        unsupervisedFeatures = arffFile.getData().copy()

        # remove label column from training data
        labelColumn = unsupervisedFeatures.columns[-1]
        unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
        y = arffFile.getData()[labelColumn]

        return unsupervisedFeatures, y

    def step1(self):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "step1"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        #step 1
        data, trueLabels = self.loadArffFile(Path(self.config["path"]))
        data = data.to_numpy()

        #step 2 plot the original dataset (picked 3 dims with highest variance)
        stds = list(zip(range(data.shape[1]), data.std(axis=0)))
        stds.sort(key=lambda x: x[1])
        dims = [i for i, _ in stds[:3]]
        Visualizer.labeledScatter3D(data[:, dims], trueLabels, path=step1ResultsFolder / f"originalScatter.png")

        #step 3
        dataMean = np.mean(data, axis=0)

        #step 4, 5, 6, 7
        pca = PCA(n_components=3, print_=True)
        reducedData = pca.fit_transform(data)

        #step 8
        Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"pcaScatter.png")

        #step 9
        reconstructedData = pca.inverse_transform(reducedData)
        Visualizer.labeledScatter3D(reconstructedData[:, dims], trueLabels, path=step1ResultsFolder / f"reconstructedScatter.png")

if __name__ == "__main__":
    args = parseArguments()
    Main(args)()
