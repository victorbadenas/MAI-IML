import sys
import numpy as np
import time
import logging
import pandas as pd
from pathlib import Path
sys.path.append('src/')
from knn import kNNAlgorithm
from computeBestKNN import *

RESULTS = Path("results/")

if __name__ == "__main__":

    datasetspd = pd.read_csv(RESULTS / 'best_models.tsv', sep='\t', index_col=0)
    set_logger(Path('log/runMultiple.log'), debug=True)
    datasets = datasetspd.index.to_list()

    logging.info(datasets)
    for dataset in datasets:
        try:
            logging.info(f"Now finding best parameters for {dataset} dataset")
            parameters = datasetspd[datasetspd.index == dataset][['param_metric', 'param_n_neighbors', 'param_p','param_voting', 'param_weights']].squeeze().to_dict()
            for k in list(parameters.keys()):
                parameters[k.replace('param_', '')] = parameters[k]
                parameters.pop(k)
            parameters['p'] = int(parameters['p'])
            tenFold = TenFoldArffFile(dataset)
            knn = kNNAlgorithm(**parameters)

            trainTimes = []
            testTimes = []
            while tenFold.loadNextFold():
                logging.info('.')
                xTrain, yTrain, xTest, yTest = getFoldData(tenFold)
                st = time.time()
                knn.fit(xTrain.to_numpy(), yTrain.to_numpy())
                trainTimes.append(time.time() - st)
                st = time.time()
                knn.predict(xTest.to_numpy())
                testTimes.append(time.time() - st)
            train_mean, train_std = np.mean(trainTimes), np.std(trainTimes)
            test_mean, test_std = np.mean(testTimes), np.std(testTimes)
            datasetspd['mean_fit_time'][dataset] = train_mean
            datasetspd['std_fit_time'][dataset] = train_std
            datasetspd['mean_score_time'][dataset] = test_mean
            datasetspd['std_score_time'][dataset] = test_std
        except Exception as e:
            logging.error(f"Exception thrown whern loading {dataset} dataset", exc_info=e)
            continue
    datasetspd.to_csv(RESULTS / 'time_recomputed.tsv', sep='\t')