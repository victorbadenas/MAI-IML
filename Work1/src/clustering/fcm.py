import numpy as np
from ..utils import convertToNumpy

"""
http://openaccess.uoc.edu/webapps/o2/bitstream/10609/59066/7/ruizjcTFG0117memoria.pdf
Page 29
"""

class FCM:
    def __init__(self, n_clusters=8):
        raise NotImplementedError
        self.reset()

    def reset(self):
        raise NotImplementedError

    def fit(self, trainData):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def fitPredict(self, data):
        self.fit(data)
        return self.predict(data)
