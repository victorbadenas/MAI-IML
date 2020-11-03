import json
import argparse
import warnings
from pathlib import Path

warnings.simplefilter(action='ignore')
from src.utils import timer
from src.dataset import ArffFile

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
        data, trueLabels = self.loadArffFile(Path(self.config["path"]))

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

if __name__ == "__main__":
    args = parseArguments()
    Main(args)()
