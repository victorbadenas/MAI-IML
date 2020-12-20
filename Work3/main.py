import argparse
import time
import json
from src.utils import timer, separateOutput
from src.dataset import TenFoldArffFile
from src.knn import kNNAlgorithm
from src.reductionKnn import reductionKnnAlgorithm
from sklearn.metrics import accuracy_score
from collections import defaultdict
from pathlib import Path


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--jsonConfig", type=Path, required=True)
    return parser.parse_args()


class Main:
    def __init__(self, args):
        self.args = args
        with open(self.args.jsonConfig, 'r') as f:
            self.config = json.load(f)
        assert 'dataset' in self.config, "No dataset has been specified."
        self.reductions = self.config.get('reduction', [])
        self.knnparameters = self.config.get('knnparameters', {})

    def __call__(self):
        self.runKnn()
        self.runReductionKnn()

    def runKnn(self):
        accuracies, efficiencies = [], []
        tenFold = TenFoldArffFile(self.config['dataset'])
        while tenFold.loadNextFold():
            xTrain, yTrain, xTest, yTest = self.getFoldData(tenFold)
            accuracy, efficiency = self.runkNNFold(xTrain, yTrain, xTest, yTest, **self.knnparameters)
            accuracies.append(accuracy)
            efficiencies.append(efficiency)
        print(f"Average Accuracy: {sum(accuracies) / len(accuracies)}")
        print(f"Average Efficiency: {sum(efficiencies) / len(efficiencies)}s")

    def runReductionKnn(self):
        accuracies, efficiencies = defaultdict(list), defaultdict(list)
        tenFold = TenFoldArffFile(self.config['dataset'])
        while tenFold.loadNextFold():
            xTrain, yTrain, xTest, yTest = self.getFoldData(tenFold)
            for reduction in self.reductions:
                accuracy, efficiency = self.runkNNReductionFold(xTrain, yTrain, xTest, yTest, **self.knnparameters, reduction=reduction)
                accuracies[reduction].append(accuracy)
                efficiencies[reduction].append(efficiency)
        for reduction in self.reductions:
            print(f"Average Accuracy for {reduction}: {sum(accuracies[reduction]) / len(accuracies[reduction])}")
            print(f"Average Efficiency for {reduction}: {sum(efficiencies[reduction]) / len(efficiencies[reduction])}s")

    @staticmethod
    def getFoldData(tenFold):
        trainData = tenFold.TrainMatrix
        testData = tenFold.TestMatrix

        # remove label column from training data and separate it
        labelColumn = trainData.columns[-1]
        xTrain = trainData.drop(labelColumn, axis=1)
        yTrain = tenFold.TrainMatrix[labelColumn]

        labelColumn = testData.columns[-1]
        xTest = testData.drop(labelColumn, axis=1)
        yTest = tenFold.TestMatrix[labelColumn]
        return xTrain, yTrain, xTest, yTest

    @staticmethod
    @separateOutput("kNN")
    @timer(print_=True)
    def runkNNFold(xTrain, yTrain, xTest, yTest, **knnparameters):
        start = time.time()
        yPred = kNNAlgorithm(**knnparameters).fit(xTrain.to_numpy(), yTrain.to_numpy()).predict(xTest.to_numpy())
        efficiency = time.time() - start
        accuracy = accuracy_score(yTest, yPred)
        print(f"Accuracy: {accuracy}")
        print(f"Efficiency: {efficiency}s")
        return accuracy, efficiency

    @staticmethod
    @separateOutput("kNNReduction")
    @timer(print_=True)
    def runkNNReductionFold(xTrain, yTrain, xTest, yTest, reduction=None, **knnparameters):
        print(f'with {reduction} reduction')
        start = time.time()
        yPred = reductionKnnAlgorithm(**knnparameters, reduction=reduction).fit(xTrain.to_numpy(), yTrain.to_numpy()).predict(xTest.to_numpy())
        efficiency = time.time() - start
        accuracy = accuracy_score(yTest, yPred)
        print(f"Accuracy: {accuracy}")
        print(f"Efficiency: {efficiency}s")
        return accuracy, efficiency


if __name__ == "__main__":
    args = parseArguments()
    Main(args)()
