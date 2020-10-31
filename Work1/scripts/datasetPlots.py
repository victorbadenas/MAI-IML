import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append("..")

from src.dataset import ArffFile

RESULTS_PATH = Path("../results")

def main():
    file_paths = [Path("../datasets/vote.arff"), Path("../datasets/adult.arff"), Path("../datasets/pen-based.arff")]
    for arfffile in file_paths:
        arffResultsFolder = RESULTS_PATH / arfffile.stem
        arffResultsFolder.mkdir(parents=True, exist_ok=True)
        arfffile = ArffFile(arfffile)
        data = arfffile.getData()
        data = data.drop(data.columns[-1], axis=1)

        arfffile.scatterPlot(ignoreLabel=True, show=False, figsize=(15, 9))
        plt.savefig(arffResultsFolder / "scatterplot.png")
        plt.close()

        plt.figure(figsize=(15, 9))
        data.boxplot()
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(arffResultsFolder / "boxplot.png")
        plt.close()

if __name__ == "__main__":
    main()