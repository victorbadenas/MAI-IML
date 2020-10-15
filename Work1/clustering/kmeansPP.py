import numpy as np
from .kmeans import KMeans
"""
from: https://en.wikipedia.org/wiki/K-means%2B%2B

k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.

The exact algorithm is as follows:

- Choose one center uniformly at random among the data points.
- For each data point x, compute D(x), the distance between x
	and the nearest center that has already been chosen.
- Choose one new data point at random as a new center, using 
	a weighted probability distribution where a point x is
	chosen with probability proportional to D(x)2.
- Repeat Steps 2 and 3 until k centers have been chosen.
- Now that the initial centers have been chosen, proceed using 
	standard k-means clustering.
"""

class KMeansPP(KMeans):
	def initializeCenters(self):
		raise NotImplementedError
