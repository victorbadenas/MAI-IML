import sys
sys.path.append(".")

import json
from pathlib import Path
from src.dataset import ArffFile

configFolderPath = Path("./configs")

for configPath in configFolderPath.glob('*.json'):
    with open(configPath) as f:
        confData = json.load(f)

    arffFilePath = confData["path"]
    arffFile = ArffFile(arffFilePath)
    arffFile.scatterPlot(ignoreLabel=True, show=True)
