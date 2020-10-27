from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from src.dataset import ArffFile

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

pass
dataNumpy = unsupervisedFeatures.to_numpy()

pca = PCA(n_components=3)

pcaData = pca.fit_transform(dataNumpy)

for label in set(clusters):
    subData = pcaData[clusters == label]
    plt.scatter(subData[:,0], subData[:,1])
plt.show()