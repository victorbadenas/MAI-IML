import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.arff import loadarff
from utils import bytesToString
from collections import Counter
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer, LabelEncoder


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
        self.path = arffPath
        data, self.metaData = loadarff(arffPath)
        self.labelEncoders = {}
        self.formatDataFrame(data)

    def formatDataFrame(self, arffData):
        self.data = pd.DataFrame(arffData)
        self.data = self.data.applymap(bytesToString) # apply type conversion to all items in DataFrame
        self.formatColumns()

    def scatterPlot(self, **kwargs):
        axes = pd.plotting.scatter_matrix(self.data, **kwargs)
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(90)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        plt.tight_layout()
        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def formatColumns(self):
        for column in self.data.columns:
            if self.data[column].dtype.kind == 'O':
                self.data[column] = self.convertStringsToInt(column, self.data[column])
            self.data[column] = self.normalizeFloatColumn(self.data[column], type="min-max")

    def convertStringsToInt(self, column, columnData):
        columnData[columnData == '?'] = Counter(columnData).most_common()[0][0]
        columnDataLabels = np.unique(columnData)
        self.labelEncoders[column] = LabelEncoder()
        self.labelEncoders[column].fit(columnDataLabels)
        columnData = self.labelEncoders[column].transform(columnData)
        return columnData

    @staticmethod
    def replaceQuestionMarksWithValue(column_data, type='mean'):
        if type == 'mean':
            column_data[column_data == '?'] = np.mean(column_data[column_data != '?'])
        elif type == 'median':
            column_data[column_data == '?'] = np.median(column_data[column_data != '?'])
        else:
            raise ValueError(f"type {type} not supported")

    @staticmethod
    def normalizeFloatColumn(data, type="min-max"):
        if type == "stardardisation":
            return scale(data)
        elif type == "mean":
            scaler = StandardScaler()
        elif type == "min-max":
            scaler = MinMaxScaler()
        elif type == "unit":
            scaler = Normalizer()
        else:
            raise ValueError(f"{type} normalization type not supported")
        data = np.array(data).reshape(-1, 1)
        scaler.fit(data)
        data = scaler.transform(data)
        return pd.Series(data.reshape(-1))

    def getData(self):
        return self.data

    def getMetaData(self):
        return self.metaData

    def getStringMapData(self):
        return self.labelEncoders

if __name__ == "__main__":
    arffFile = ArffFile(Path("./datasets/adult.arff"))
    print(arffFile.getData().head())
    arffFile.scatterPlot(figsize=(10, 10))
