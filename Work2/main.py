import json
import argparse
import warnings
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import IncrementalPCA

from src.utils import timer
from src.dataset import ArffFile
from src.visualize import Visualizer
from src.pca import PCA
from src.clustering import KMeans
warnings.simplefilter(action='ignore')


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--jsonConfig", type=Path, required=True)
    return parser.parse_args()


class Main:
    def __init__(self, args):
        self.args = args
        with open(self.args.jsonConfig, 'r') as f:
            self.config = json.load(f)

    def __call__(self):
        self.step1()
        self.step3()
        self.step4()

    @staticmethod
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

    @timer(print_=True)
    def step1(self):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "step1"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        # step 1
        data, trueLabels = self.loadArffFile(Path(self.config["path"]))
        data = data.to_numpy()

        # step 2 plot the original dataset (picked 3 dims with highest variance)
        stds = list(zip(range(data.shape[1]), data.std(axis=0)))
        stds.sort(key=lambda x: x[1])
        dims = [i for i, _ in stds[:3]]
        Visualizer.labeledScatter3D(data[:, dims], trueLabels, path=step1ResultsFolder / f"originalScatter.png")

        # step 3
        dataMean = np.mean(data, axis=0)
        print(dataMean)

        # step 4, 5, 6, 7
        pca = PCA(n_components=3, print_=True)
        reducedData = pca.fit_transform(data)

        # step 8
        Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"pcaScatter.png")

        # step 9
        reconstructedData = pca.inverse_transform(reducedData)
        Visualizer.labeledScatter3D(reconstructedData[:, dims], trueLabels, path=step1ResultsFolder / f"reconstructedScatter.png")

    @timer(print_=True)
    def step3(self):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "step3"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        data, trueLabels = self.loadArffFile(Path(self.config["path"]))
        data = data.to_numpy()

        pca = skPCA(3)
        ipca = IncrementalPCA(3)
        reducedData = pca.fit_transform(data)
        iReducedData = ipca.fit_transform(data)

        Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"pcaScatter.png")
        Visualizer.labeledScatter3D(iReducedData, trueLabels, path=step1ResultsFolder / f"ipcaScatter.png")

    @timer(print_=True)
    def step4(self):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "step4"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        data, trueLabels = self.loadArffFile(Path(self.config["path"]))
        data = data.to_numpy()

        pca = skPCA(3)
        reducedData = pca.fit_transform(data)

        kMeans = KMeans(**self.config["parameters"]["kMeans"])
        predictedLabels = kMeans.fitPredict(reducedData)

        Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"gsScatter.png")
        Visualizer.labeledScatter3D(reducedData, predictedLabels, path=step1ResultsFolder / f"kmeansScatter.png")


if __name__ == "__main__":
    args = parseArguments()
    Main(args)()
