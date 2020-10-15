import numpy as np

class KMeans:
	def __init__(self):
		self.initializeCenters()
	
	def initializeCenters(self):
		raise NotImplementedError

	def __call__(self):
		raise NotImplementedError
