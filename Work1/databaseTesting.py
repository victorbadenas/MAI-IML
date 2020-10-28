from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from src.dataset import ArffFile
import matplotlib.pyplot as plt

'''
https://medium.com/@rodrigodutcosky/creating-a-3d-scatter-plot-from-your-clustered-data-with-plotly-843c20b78799
'''
def runDBSCAN(data):
    clustering = DBSCAN(n_jobs=-1, eps=.75)
    labels = clustering.fit_predict(data)
    return labels

arffFile = ArffFile("datasets/adult.arff")
unsupervisedFeatures = arffFile.getData().copy()

labelColumn = unsupervisedFeatures.columns[-1]
unsupervisedFeatures = unsupervisedFeatures.drop(labelColumn, axis=1)
y = arffFile.getData()[labelColumn]

clusters = runDBSCAN(unsupervisedFeatures)

dataNumpy = unsupervisedFeatures.to_numpy()

pca = PCA(n_components=3)

pcaData = pca.fit_transform(dataNumpy)

for label in set(clusters):
    subData = pcaData[clusters == label]
    plt.scatter(subData[:,0], subData[:,1])
plt.show()

# # 3D PLOT IDEA
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
 
# # Dataset
# df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })
 
# # plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)
# ax.view_init(30, 185)
# plt.show()
