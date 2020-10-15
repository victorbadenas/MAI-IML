import numpy as np
"""
https://en.wikipedia.org/wiki/K-means_clustering
"""
class KMeans:
    def __init__(self):
        self.initializeCenters()
    
    def initializeCenters(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
