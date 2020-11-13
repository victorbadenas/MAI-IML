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
from src.kmeans import KMeans
warnings.simplefilter(action='ignore')

N_COMPONENTS = 3

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
        # step 1
        data, trueLabels = self.loadArffFile(Path(self.config["path"]))
        data = data.to_numpy()

        reducedDataMyPCA = self.phase1(data, trueLabels)
        reducedDataskPCA, reducedDataiPCA = self.phase3(data, trueLabels)
        self.phase4(data, reducedDataMyPCA, reducedDataskPCA, reducedDataiPCA, trueLabels)

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
    def phase1(self, data, trueLabels):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "phase1"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        # step 2 plot the original dataset (picked N_COMPONENTS dims with highest variance)
        stds = data.std(axis=0)
        dims = np.argsort(stds)[::-1][:N_COMPONENTS]
        Visualizer.labeledScatter3D(data[:, dims], trueLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_originalScatter.png")

        # step 3
        dataMean = np.mean(data, axis=0)
        print(f"Original data mean: {dataMean}")

        # step 4, 5, 6, 7
        pca = PCA(n_components=N_COMPONENTS, print_=True)
        reducedData = pca.fit_transform(data)

        # step 8
        Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_pcaScatter.png")

        # step 9
        reconstructedData = pca.inverse_transform(reducedData)
        Visualizer.labeledScatter3D(reconstructedData[:, dims], trueLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_reconstructedScatter.png")
        return reducedData


    @timer(print_=True)
    def phase3(self, data, trueLabels):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "phase3"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        pca = skPCA(N_COMPONENTS)
        ipca = IncrementalPCA(N_COMPONENTS)
        reducedData = pca.fit_transform(data)
        iReducedData = ipca.fit_transform(data)

        Visualizer.labeledScatter3D(reducedData, trueLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_pcaScatter.png")
        Visualizer.labeledScatter3D(iReducedData, trueLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_ipcaScatter.png")
        return reducedData, iReducedData


    @timer(print_=True)
    def phase4(self, originalData, reducedDataMyPCA, reducedDataskPCA, reducedDataiPCA, trueLabels):
        step1ResultsFolder = Path(self.config["resultsDir"]) / "phase4"
        step1ResultsFolder.mkdir(exist_ok=True, parents=True)

        for title, data in zip(["originalData", "myPCA", "sklearnPCA", "incrementalPCA"], 
                               [originalData, reducedDataMyPCA, reducedDataskPCA, reducedDataiPCA]):
            kMeans = KMeans(**self.config["parameters"]["kMeans"])
            predictedLabels = kMeans.fitPredict(data)
            Visualizer.labeledScatter3D(data, trueLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_{title}_gsScatter.png")
            Visualizer.labeledScatter3D(data, predictedLabels, path=step1ResultsFolder / f"{N_COMPONENTS}_dims_{title}_kmeansScatter.png")


if __name__ == "__main__":
    args = parseArguments()
    Main(args)()
