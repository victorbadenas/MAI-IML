import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.arff import loadarff
from .utils import bytesToString
from collections import Counter
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OneHotEncoder


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
        maps: dictionary with mapping information from string to int or OneHotVector

    """
    def __init__(self, arffPath, stringConversion='int', floatNormalization="min-max"):
        self.path = arffPath
        self.stringConversion = stringConversion
        self.floatNormalization = floatNormalization
        data, self.metaData = loadarff(arffPath)
        self.labelEncoders = {}
        self.formatDataFrame(data)

    def formatDataFrame(self, arffData):
        self.data = pd.DataFrame(arffData)
        self.data = self.data.applymap(bytesToString) # apply type conversion to all items in DataFrame
        self.formatColumns()

    def scatterPlot(self, ignoreLabel=False, **kwargs):
        if ignoreLabel:
            axes = pd.plotting.scatter_matrix(self.data.copy().drop('class', axis=1), **kwargs)
        else:
            axes = pd.plotting.scatter_matrix(self.data, **kwargs)
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(90)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        plt.tight_layout()
        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def formatColumns(self):
        columnNames = self.data.columns.to_list()
        for column in self.data.columns[:-1].to_list():
            columnData = self.data[column].copy()
            if columnData.dtype.kind == 'O':
                if self.stringConversion == 'int':
                    self.data[column] = self.convertStringsToInt(column, columnData).astype(np.int)
                elif self.stringConversion == 'onehot':
                    ohDataFrame = self.convertStringToOH(column, columnData)
                    self.data = self.data.drop(column, axis=1)
                    columnPosition = columnNames.index(column)
                    columnsOHDataFrame = ohDataFrame.columns.to_list()
                    columnNames[columnPosition] = columnsOHDataFrame[0]
                    for i in range(1, len(columnsOHDataFrame)):
                        columnNames.insert(columnPosition + i, columnsOHDataFrame[i])
                    self.data = self.data.join(ohDataFrame, how='outer')[columnNames]
                else:
                    raise NotImplementedError(f"{self.stringConversion} is not a valid value for stringConversion")

        gsColumn = columnNames[-1]
        self.data[gsColumn] = self.convertStringsToInt(gsColumn, self.data[gsColumn]).astype(np.int)

        for column in self.data.columns[:-1]:
            columnData = self.data[column].copy()
            self.data[column] = self.normalizeFloatColumn(columnData, type=self.floatNormalization)

    def convertStringsToInt(self, column, columnData):
        columnData[columnData == '?'] = Counter(columnData[columnData != '?']).most_common()[0][0]
        columnDataLabels = np.unique(columnData)
        self.labelEncoders[column] = LabelEncoder()
        self.labelEncoders[column].fit(columnDataLabels)
        columnData = self.labelEncoders[column].transform(columnData)
        return columnData

    def convertStringToOH(self, columnName, columnData):
        columnData[columnData == '?'] = Counter(columnData[columnData != '?']).most_common()[0][0]
        columnDataOH = pd.get_dummies(columnData)
        mapper = {columnValue: f"{columnName}_{columnValue}" for columnValue in columnDataOH.columns}
        columnDataOH = columnDataOH.rename(columns=mapper)
        return columnDataOH

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
