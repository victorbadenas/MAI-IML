import sys
sys.path.append(".")

import time
import glob
from pathlib import Path
from src.dataset import TenFoldArffFile
from src.knn import kNNAlgorithm, COSINE, EUCLIDEAN, VOTING, WEIGHTS, UNIFORM
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def runDataset(dataset, parameters):
    accuracies, efficiencies = [], []
    tenFold = TenFoldArffFile(dataset)
    while tenFold.loadNextFold():
        xTrain, yTrain, xTest, yTest = getFoldData(tenFold)
        accuracy, efficiency = runkNN(xTrain, yTrain, xTest, yTest, parameters)
        accuracies.append(accuracy)
        efficiencies.append(efficiency)
    return sum(accuracies) / len(accuracies), sum(efficiencies) / len(efficiencies)

def getFoldData(tenFold):
    trainData = tenFold.TrainMatrix
    testData = tenFold.TestMatrix

    labelColumn = trainData.columns[-1]
    xTrain = trainData.drop(labelColumn, axis=1)
    yTrain = tenFold.TrainMatrix[labelColumn]

    labelColumn = testData.columns[-1]
    xTest = testData.drop(labelColumn, axis=1)
    yTest = tenFold.TestMatrix[labelColumn]
    return xTrain, yTrain, xTest, yTest

def runkNN(xTrain, yTrain, xTest, yTest, parameters):
    #start = time.time()
    #yPred = kNNAlgorithm().fit(xTrain.to_numpy(), yTrain.to_numpy()).predict(xTest.to_numpy())
    #efficiency = time.time() - start
    #accuracy = accuracy_score(yTest, yPred)
    #return accuracy, efficiency
    clf = GridSearchCV(kNNAlgorithm(), parameters, scoring='accuracy')
    clf.fit(xTrain.to_numpy(), yTrain.to_numpy())

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("\nDetailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = yTest, clf.predict(xTest.to_numpy())
    print(classification_report(y_true, y_pred))
    return 0, 0


# NOTE: Sembla que si computo el GridSearch incloent tots els testos triga mil anys i per això està comentat
# TODO: For the evaluation, you will use a T-Test or another statistical method (llegir paper T-Test)
# NOTE 2: No sé si hi ha alguna manera d'extreure la metrica de 'temps' dins el GridSearch
if __name__ == "__main__":
    #datasets = [Path(path).stem for path in glob.glob("10fdatasets/*")]
    datasets = ["autos"]
    #parameters = [{'n_neighbors': [1, 3, 5, 7], 'weights': WEIGHTS, 'voting': VOTING},
    #              {'n_neighbors': [1, 3, 5, 7], 'metric': [EUCLIDEAN], 'weights': WEIGHTS, 'voting': VOTING, 'p': [2]},
    #              {'n_neighbors': [1, 3, 5, 7], 'metric': [COSINE], 'weights': WEIGHTS, 'voting': VOTING}]]
    parameters = [{'n_neighbors': [1, 3, 5, 7], 'metric': [EUCLIDEAN], 'weights': [UNIFORM], 'voting': VOTING, 'p': [2]}]
    accuracies, efficiencies = {}, {}
    for dataset in datasets:
        accuracy, efficiency = runDataset(dataset, parameters)
        accuracies[dataset] = accuracy
        efficiencies[dataset] = efficiency