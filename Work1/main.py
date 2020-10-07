import argparse
import warnings
warnings.simplefilter(action='ignore')
from dataset import ArffFile
from pathlib import Path
from sklearn.cluster import DBSCAN


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--arffFolder", type=Path, default="./datasets/")
    return parser.parse_args()

class Main:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        arffFilesPaths = list(self.args.arffFolder.glob("*.arff"))
        # arffFilesData = [ArffFile(arffFilePath) for arffFilePath in arffFilesPaths]
        arffFile = ArffFile(arffFilesPaths[0])
        clustering = DBSCAN(n_jobs=-1)
        labels = clustering.fit_predict(arffFile.getData())
        print(labels)
        return

if __name__ == "__main__":
    args = parseArguments()
    Main(args)()