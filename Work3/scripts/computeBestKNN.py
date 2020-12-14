import sys
sys.path.append(".")
import time
import glob
import logging
import numpy as np
from pathlib import Path
from src.dataset import TenFoldArffFile
from src.knn import kNNAlgorithm, DISTANCE_METRICS, COSINE, EUCLIDEAN, VOTING, WEIGHTS, UNIFORM, CORRELATION
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from itertools import product
import pandas as pd
from src.utils import getSizeOfObject



def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)


def computePredefinedSplit(dataset, parameters):
    tenFold = TenFoldArffFile(dataset)
    X = None
    Y = None
    foldIdx = 0
    while tenFold.loadNextFold():
        xTrain, yTrain, xTest, yTest = getFoldData(tenFold)
        if X is None:
            X = np.concatenate([xTrain, xTest])
            Y = np.concatenate([yTrain, yTest])
            indexes = np.full(X.shape[0], -1)
        xTrain = xTrain.to_numpy()
        xTest = xTest.to_numpy()
        for item in xTest:
            index = np.where((X == item).all(axis=1))[0]
            indexes[index] = foldIdx
        pass
        foldIdx += 1
    return X, Y, PredefinedSplit(indexes)


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
    # start = time.time()
    # yPred = kNNAlgorithm().fit(xTrain.to_numpy(), yTrain.to_numpy()).predict(xTest.to_numpy())
    # efficiency = time.time() - start
    # accuracy = accuracy_score(yTest, yPred)
    # return accuracy, efficiency
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


def loopOverParameters(dataset, parameters):
    dataframeColumns = ['paramIndex', 'fold', 'accuracy', 'trainTime', 'testTime', 'storage', 'parameters']
    results = [pd.DataFrame(columns=dataframeColumns)]*len(parameters)
    for setupIdx, parameterSetup in enumerate(parameters):
        logging.info(f"Now evaluating parameter setup: {parameterSetup}")
        parameterKeys = parameterSetup.keys()
        parameterList = [p for _, p in parameterSetup.items()]
        parameterCombinations = list(product(*parameterList))

        for paramIdx, parameterCombination in enumerate(parameterCombinations):
            tenFold = TenFoldArffFile(dataset)
            parameterdict = dict(zip(parameterKeys, parameterCombination))
            logging.info(f"Now evaluating combination {paramIdx+1}/{len(parameterCombinations)}: {parameterdict}")

            foldIdx = 0
            while tenFold.loadNextFold():
                xTrain, yTrain, xTest, yTest = getFoldData(tenFold)

                st = time.time()
                knn = kNNAlgorithm(**parameterdict).fit(xTrain.to_numpy(), yTrain.to_numpy())
                trainTime = time.time() - st

                st = time.time()
                pred = knn.predict(xTest)
                testTime = time.time() - st

                accuracy = np.average(pred == yTest)
                storage = getSizeOfObject(knn)

                foldResults = pd.DataFrame(np.array([[paramIdx, foldIdx, accuracy, trainTime, testTime, storage, parameterdict]]), columns=dataframeColumns)
                results[setupIdx] = results[setupIdx].append(foldResults, ignore_index=True)

                foldIdx += 1

    return results


# NOTE: Sembla que si computo el GridSearch incloent tots els testos triga mil anys i per això està comentat
# TODO: For the evaluation, you will use a T-Test or another statistical method (llegir paper T-Test)
# NOTE 2: No sé si hi ha alguna manera d'extreure la metrica de 'temps' dins el GridSearch
if __name__ == "__main__":
    datasets = [Path(path).stem for path in glob.glob("10fdatasets/*")]

    set_logger(Path('log/bestKnn.log'), debug=True)
    resultsPath = Path('./results/best_knn')
    resultsPath.mkdir(parents=True, exist_ok=True)
    logging.info(datasets)
    # parameters = [{'n_neighbors': [1, 3, 5, 7], 'weights': WEIGHTS, 'voting': VOTING},
    #               {'n_neighbors': [1, 3, 5, 7], 'metric': [EUCLIDEAN], 'weights': WEIGHTS, 'voting': VOTING, 'p': [2]},
    #               {'n_neighbors': [1, 3, 5, 7], 'metric': [COSINE], 'weights': WEIGHTS, 'voting': VOTING}]]
    # parameters = [{'n_neighbors': [1, 3, 5, 7], 'metric': [EUCLIDEAN], 'weights': [UNIFORM], 'voting': VOTING, 'p': [2]}]
    parameters = [{'n_neighbors': [1, 3, 5, 7], 'metric': DISTANCE_METRICS, 'weights': WEIGHTS, 'voting': VOTING, 'p': [2]}]
    accuracies, efficiencies = {}, {}
    for dataset in datasets:
        if dataset == 'connect-4':
            continue
        try:
            logging.info(f"Now finding best parameters for {dataset} dataset")
            fullDataset, fullLabels, predefinedSplit = computePredefinedSplit(dataset, parameters)
            logging.info(f"datset loaded with {fullDataset.shape} shape and predefined split computed")
            gs = GridSearchCV(
                kNNAlgorithm(),
                parameters,
                scoring='accuracy',
                cv=predefinedSplit,
                n_jobs=-1,
                refit=True,
                verbose=1)
            gs.fit(fullDataset, fullLabels)
            logging.info("best model found")
            knn = gs.best_estimator_
            memory = getSizeOfObject(knn)
            logging.info(f"memory used by best model: {memory/1024}kb")
            resultsDf = pd.DataFrame(gs.cv_results_)
            resultsDf.to_csv(resultsPath / (dataset + '.tsv'), sep='\t')
            logging.info(f"results saved to {resultsPath / (dataset + '.tsv')}")
        except Exception as e:
            logging.error(f"Exception thrown whern loading {dataset} dataset")
            logging.error(e)
            continue

