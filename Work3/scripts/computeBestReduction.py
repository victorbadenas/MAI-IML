import sys
sys.path.append(".")
import time
import glob
import logging
import numpy as np
from pathlib import Path
from src.dataset import TenFoldArffFile
from src.knn import *
from src.reductionKnn import reductionKnnAlgorithm, REDUCTION_METHODS
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from itertools import product
import pandas as pd
from src.utils import getSizeOfObject
from collections import defaultdict


def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)


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


def loopOverParameters(dataset, parameters, algorithm):
    resultscolumns = []
    results = dict()

    parameterKeys = parameters.keys()
    parameterList = [p for _, p in parameters.items()]
    parameterCombinations = list(product(*parameterList))

    logging.info(f"evaluating {algorithm.__name__} with dataset {dataset} and {len(parameterCombinations)} parameter combinations")

    
    for paramIdx, combination in enumerate(parameterCombinations):
        parameterdict = dict(zip(parameterKeys, combination))
        result_dict = parameterdict.copy()

        logging.info(f"Now evaluating combination {paramIdx+1}/{len(parameterCombinations)}: {parameterdict}")

        fit_accuracies = []
        accuracies = []
        fit_times = []
        score_times = []
        storages = []

        tenFold = TenFoldArffFile(dataset)
        foldIdx = 0
        while tenFold.loadNextFold():
            xTrain, yTrain, xTest, yTest = getFoldData(tenFold)

            st = time.time()
            knn = algorithm(**parameterdict)
            trainLabels = knn.fit_predict(xTrain.to_numpy(), yTrain.to_numpy())
            trainTime = time.time() - st
            trainAccuracy = np.average(trainLabels == yTrain)
            st = time.time()
            pred = knn.predict(xTest.to_numpy())
            testTime = time.time() - st

            accuracy = np.average(pred == yTest)
            storage = getSizeOfObject(knn)

            fit_accuracies.append(trainAccuracy)
            accuracies.append(accuracy)
            fit_times.append(trainTime)
            score_times.append(testTime)
            storages.append(storage)

            foldIdx += 1

        for i, score in enumerate(fit_accuracies):
            result_dict[f'train_score_fold{i}'] = fit_accuracies[i]

        for i, score in enumerate(accuracies):
            result_dict[f'test_score_fold{i}'] = accuracies[i]

        result_dict['mean_fit_time'] = np.mean(fit_times)
        result_dict['std_fit_time'] = np.std(fit_times)
        result_dict['mean_score_time'] = np.mean(score_times)
        result_dict['std_score_time'] = np.std(score_times)
        result_dict['storage_mean'] = np.mean(storages)
        result_dict['mean_fit_score'] = np.mean(fit_accuracies)
        result_dict['std_fit_score'] = np.std(fit_accuracies)
        result_dict['mean_score'] = np.mean(accuracies)
        result_dict['std_score'] = np.std(accuracies)

        logging.info(f"combination {paramIdx+1}/{len(parameterCombinations)}: fit_time={result_dict['mean_fit_time']:4f}, score_time={result_dict['mean_score_time']:4f}, accuracy={np.mean(accuracies):.4f}, storage={int(result_dict['storage_mean'])}")
        results[paramIdx] = result_dict
    return pd.DataFrame.from_dict(results, orient="index")

if __name__ == "__main__":
    datasets = ['hypothyroid', 'splice']
    bestKNNparameters = [
        {'metric': ['cosine'], 'n_neighbors':[5], 'p':[2], 'voting': ['majority'], 'weights': ['mutual_info']},
        {'metric': ['minkowski2'], 'n_neighbors':[1], 'p':[2], 'voting': ['sheppards'], 'weights': ['mutual_info']}
    ]

    set_logger(Path('log/bestReduction.log'), debug=True)
    resultsPath = Path('./results/best_reduction')
    resultsPath.mkdir(parents=True, exist_ok=True)
    logging.info(datasets)

    for dataset, parameters in zip(datasets, bestKNNparameters):
        try:
            parameters['reduction'] = [None]
            results = loopOverParameters(dataset, parameters, reductionKnnAlgorithm)
            results.to_csv(resultsPath / f'{dataset}.tsv', sep='\t')
        except Exception as e:
            logging.error(f"Exception thrown when evaluating {dataset} dataset", exc_info=e)
            continue