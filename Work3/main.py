import argparse
import time
#import numpy as np
#import pandas as pd
from src.utils import timer, separateOutput
from src.dataset import TenFoldArffFile
from src.knn import kNNAlgorithm
from sklearn.metrics import accuracy_score


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    return parser.parse_args()


class Main:
    def __init__(self, args):
        self.args = args

    def __call__(self, *args, **kwargs):
        accuracies, efficiencies = [], []
        tenFold = TenFoldArffFile(self.args.dataset)
        while tenFold.loadNextFold():
            xTrain, yTrain, xTest, yTest = self.getFoldData(tenFold)
            accuracy, efficiency = self.runkNN(xTrain, yTrain, xTest, yTest)
            accuracies.append(accuracy)
            efficiencies.append(efficiency)
        print(f"Average Accuracy: {sum(accuracies) / len(accuracies)}")
        print(f"Average Efficiency: {sum(efficiencies) / len(efficiencies)}s")

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
    def runkNN(xTrain, yTrain, xTest, yTest):
        start = time.time()
        yPred = kNNAlgorithm().fit(xTrain.to_numpy(), yTrain.to_numpy()).predict(xTest.to_numpy())
        efficiency = time.time() - start
        accuracy = accuracy_score(yTest, yPred)
        print(f"Accuracy: {accuracy}")
        print(f"Efficiency: {efficiency}s")
        return accuracy, efficiency


if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    Main(args)()
