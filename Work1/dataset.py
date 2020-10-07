import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.arff import loadarff
from utils import bytesToString


class ArffFile:
    """
    Class responsible of loading and formatting the arff file data

    input:
        args:
            arffPath: Path object with information on the path to arffFile

    attributes:
        path: path to arff file
        data: pd.DataFrame with arff data in float values (normalised in range [0, 1])
        metaData: arff metadata information
        maps: dictionary with mapping information from string to int

    """
    def __init__(self, arffPath):
        super(ArffFile, self).__init__()
        self.path = arffPath
        data, self.metaData = loadarff(arffPath)
        self.maps = {}
        self.formatDataFrame(data)

    def formatDataFrame(self, arffData):
        self.data = pd.DataFrame(arffData)
        self.data = self.data.applymap(bytesToString)
        self.formatColumns()

    def scatterPlot(self, **kwargs):
        plt.figure(figsize=(15,15))
        pd.plotting.scatter_matrix(self.data, **kwargs)
        plt.show()

    def formatColumns(self):
        for column in self.data.columns:
            if self.data[column].dtype.kind == 'O':
                self.data[column] = self.convertStringsToInt(column, self.data[column])
            elif self.data[column].dtype.kind == 'f':
                self.data[column] = self.normalizeFloatColumn(self.data[column])

    def convertStringsToInt(self, column, column_data):
        columnMap = self.createMap(column, column_data)
        self.maps[column] = columnMap
        column_data = column_data.apply(lambda x: columnMap[x]/columnMap['max_range'] if x!= '?' else x)
        column_data[column_data == '?'] = np.mean(column_data[column_data != '?'])
        return column_data

    @staticmethod
    def normalizeFloatColumn(data):
        return data / max(data)

    def createMap(self, column, columnData):
        columnData = set(columnData)
        if '?' in columnData:
            columnData.remove('?')
        columnData = sorted(columnData)
        max_range = len(columnData) - 1
        int_labels = list(range(len(columnData)))
        map_ = dict(zip(columnData, int_labels))
        map_["max_range"] = max_range
        return map_

    def getData(self):
        return self.data

    def getMetaData(self):
        return self.metaData

    def getStringMapData(self):
        return self.maps

if __name__ == "__main__":
    arffFile = ArffFile(Path("./datasets/adult.arff"))
    print(arffFile.getData().head())
