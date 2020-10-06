import argparse
from dataset import ArffFile
from pathlib import Path


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--arffFolder", type=Path, default="./datasets/")
    return parser.parse_args()

class Main:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        arffFilesPaths = list(self.args.arffFolder.glob("*.arff"))
        arffFilesData = [ArffFile(arffFilePath) for arffFilePath in arffFilesPaths]


if __name__ == "__main__":
    args = parseArguments()
    Main(args)()