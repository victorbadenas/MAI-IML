import numpy as np
from .kmeans import KMeans
"""
https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2

there is a python implementation there, but we shouldn't use it
"""
class BisectingKMeans:
    pass

'''

Input: number of clusters K, data points x1,...,x_n
Output: K cluster centers, c_1,...,c_k 
1. Pick a cluster to split.
2. Find 2 sub-clusters using the basic k-Means algorithm (Bisecting step)
3. Repeat step 2, the bisecting step, for ITER times and take the split that produces the clustering with the highest overall similarity.
4. Repeat steps 1, 2 and 3 until the desired number of clusters is reached.

import numpy as np
from .kmeans import KMeans

class BisectingKMeans:

    def l2dist(points, center, axis=0):
        dist = (points - center)**2
        dist = np.sum(dist, axis=1)
        return np.sqrt(dist)
        
    def __init__(self, n_clusters=8, n_init=10):
        self.numberOfClusters = n_clusters
        self.nInit = n_init
        self.centers = None
    
    def fit(self, trainData):
        trainData = convertToNumpy(trainData)
        self._initializeCenters(trainData)
        for _ in range(self.numberOfClusters - 1):
            minCentroidDistance = self._predictClusters(trainData)
            newCenter = self._computeNewCenter(trainData, minCentroidDistance)
            self._updateCenters(newCenter)
        return self  
      
    def get_centroids(self):
        return self.centers    
    
    def bisectingKmeans(traindata, n_clusters=2, max_iter=100, verbose=False )
        #Initial Cluster
        data=convertToNumpy(trainData)
        
        cluster=[]
        cluster.append(data)
        #loop until n_cluster iteration occur
        While len(cluster)<n_clusters:
            #Obtainin SSE and selecting max error
            SSE_values=[]
            for x in cluster:
                SSE=l2dist(cluster[x], get_centroids[x], axis=0)
                SSE_values.append(SSE)
            maxvalue = max(SSE_values)
            maxpos = list.index(maxvalue)
            
            #Running Kmeans for selected cluster
            result_cluster = KMeans(cluster[maxpos],n_clusters=2, verbose=self.args.verbose)
            
            #Removing old cluster that would be divided into two
            cluster.pop(maxpos)           
            cluster.append(result_cluster)
            
        return result_cluster

'''


