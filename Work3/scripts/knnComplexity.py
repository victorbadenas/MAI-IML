import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy import stats
import sys
import time
sys.path.append('src/')
from utils import getSizeOfObject
from knn import kNNAlgorithm

MAX_DATA_POINTS = 20000
MAX_DATA_POINTS_SCIPY = 1000
MAX_FEATURES = 40

def plotModelTrial(trainData, testData, trainLabels, testLabels, classgs):
    plt.figure(figsize=(15, 9))
    for label, c in zip(np.unique(trainLabels), 'rgby'):
        subData = trainData[trainLabels == label]
        subNewData = testData[testLabels == label]
        plt.scatter(subData[:, 0], subData[:, 1], c=c, marker='+')
        plt.scatter(subNewData[:, 0], subNewData[:, 1], c=c, marker='x')
    # plt.scatter(classgs[:, 0], classgs[:, 1], c='k', marker='o')
    plt.vlines(0.5, 0, 1, colors='k', linestyles='dashed')
    plt.hlines(0.5, 0, 1, colors='k', linestyles='dashed')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([i/4 for i in range(5)])
    plt.yticks([i/4 for i in range(5)])
    plt.grid('on')


# DATA POINTS RELATION (TIME, SPACE)
def n_data_complexity():
    classgs = np.array([(0.75, 0.75),(0.25, 0.25),(0.75, 0.25),(0.25, 0.75)])

    n_points_trials = list(range(1000, MAX_DATA_POINTS + 1, 1000))
    train_times = list()
    test_times = list()
    storages = list()

    for n_points in n_points_trials:
        test_points = n_points // 10
        n_points_per_class = n_points // 4

        data = list()
        labels = list()

        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.75, 0.75))
        labels.append(np.zeros((n_points_per_class,)))
        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.25, 0.25))
        labels.append(np.full((n_points_per_class,), 1))
        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.75, 0.25))
        labels.append(np.full((n_points_per_class,), 2))
        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.25, 0.75))
        labels.append(np.full((n_points_per_class,), 3))

        data = np.vstack(data)
        labels = np.concatenate(labels)

        test_idx = np.random.choice(data.shape[0], test_points, replace=False)
        testData = data[test_idx]
        testLabels = labels[test_idx]
        trainData = np.delete(data, test_idx, axis=0)
        trainLabels = np.delete(labels, test_idx, axis=0)

        print(f"train dataset size: {trainData.shape}, test dataset size: {testData.shape}")

        st = time.time()

        knn = kNNAlgorithm(metric='minkowski2', voting='idw', weights='mutual_info', method='mat').fit(trainData, trainLabels)
        fit_time = time.time() - st

        st = time.time()
        pred_labels = knn.predict(testData)
        predict_time = time.time() - st
        
        storage = getSizeOfObject(knn)/1024

        train_times.append(fit_time)
        test_times.append(predict_time)
        storages.append(storage)

        print(f"Has runned with: train time={fit_time:5f}s test time={predict_time:5f}s storage={storage}kB")

    plt.figure(figsize=(15, 9))
    plt.subplot(311)
    plt.plot(n_points_trials, train_times)
    plt.grid('on')
    plt.xlabel('dataset size (samples)')
    plt.ylabel('fit_time (s)')

    plt.subplot(312)
    plt.plot(n_points_trials, test_times)
    plt.grid('on')
    plt.xlabel('dataset size (samples)')
    plt.ylabel('predict_time (s)')

    plt.subplot(313)
    plt.plot(n_points_trials, storages)
    plt.grid('on')
    plt.xlabel('dataset size (samples)')
    plt.ylabel('storage (kB)')

    plt.tight_layout()
    outpath = Path('results/performance')
    outpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath / 'n_dataset_o.png', dpi=300)
    plt.show()
    plt.close()


## NUM FEATURES (TIME, SPACE)
def n_features_complexity():
    n_features_trials = list(range(2, MAX_FEATURES, 1))
    train_times = list()
    test_times = list()
    storages = list()

    n_points = 5000
    test_points = n_points // 10
    n_points_per_class = n_points // 4

    for n_f in n_features_trials:
        classgs = np.array([
            ((0.75, 0.75) + tuple([0.75]*(n_f-2))),
            ((0.25, 0.25) + tuple([0.75]*(n_f-2))),
            ((0.75, 0.25) + tuple([0.75]*(n_f-2))),
            ((0.25, 0.75) + tuple([0.75]*(n_f-2)))
        ])

        data = []
        labels = []
        data.append(np.random.rand(n_points_per_class, n_f)/2 - 0.25 + ((0.75, 0.75) + tuple([0.75]*(n_f-2))))
        labels.append(np.zeros((n_points_per_class,)))
        data.append(np.random.rand(n_points_per_class, n_f)/2 - 0.25 + ((0.25, 0.25) + tuple([0.75]*(n_f-2))))
        labels.append(np.full((n_points_per_class,), 1))
        data.append(np.random.rand(n_points_per_class, n_f)/2 - 0.25 + ((0.75, 0.25) + tuple([0.75]*(n_f-2))))
        labels.append(np.full((n_points_per_class,), 2))
        data.append(np.random.rand(n_points_per_class, n_f)/2 - 0.25 + ((0.25, 0.75) + tuple([0.75]*(n_f-2))))
        labels.append(np.full((n_points_per_class,), 3))
        data = np.vstack(data)
        labels = np.concatenate(labels)

        test_idx = np.random.choice(data.shape[0], test_points, replace=False)
        testData = data[test_idx]
        testLabels = labels[test_idx]
        trainData = np.delete(data, test_idx, axis=0)
        trainLabels = np.delete(labels, test_idx, axis=0)

        print(f"train dataset size: {trainData.shape}, test dataset size: {testData.shape}")

        st = time.time()

        knn = kNNAlgorithm(metric='minkowski2', voting='idw', weights='mutual_info', method='mat').fit(trainData, trainLabels)
        fit_time = time.time() - st

        st = time.time()
        pred_labels = knn.predict(testData)
        predict_time = time.time() - st
        
        storage = getSizeOfObject(knn)/1024

        train_times.append(fit_time)
        test_times.append(predict_time)
        storages.append(storage)

        print(f"Has runned with: train time={fit_time:5f}s test time={predict_time:5f}s storage={storage}kB")

    plt.figure(figsize=(15, 9))
    plt.subplot(311)
    plt.plot(n_features_trials, train_times)
    plt.grid('on')
    plt.xlabel('feature size')
    plt.ylabel('fit_time (s)')

    plt.subplot(312)
    plt.plot(n_features_trials, test_times)
    plt.grid('on')
    plt.xlabel('feature size')
    plt.ylabel('predict_time (s)')

    plt.subplot(313)
    plt.plot(n_features_trials, storages)
    plt.grid('on')
    plt.xlabel('feature size')
    plt.ylabel('storage (kB)')

    plt.tight_layout()
    outpath = Path('results/performance')
    outpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath / 'n_features_o.png', dpi=300)
    plt.show()
    plt.close()


def scipy_vs_mat():
    classgs = np.array([(0.75, 0.75),(0.25, 0.25),(0.75, 0.25),(0.25, 0.75)])

    n_points_trials = list(range(100, MAX_DATA_POINTS_SCIPY + 1, 100))
    train_times = list()
    test_times = list()
    storages = list()

    mat_times = []
    scipy_times = []
    errs = []

    for n_points in n_points_trials:
        test_points = n_points // 10
        n_points_per_class = n_points // 4

        data = list()
        labels = list()

        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.75, 0.75))
        labels.append(np.zeros((n_points_per_class,)))
        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.25, 0.25))
        labels.append(np.full((n_points_per_class,), 1))
        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.75, 0.25))
        labels.append(np.full((n_points_per_class,), 2))
        data.append(np.random.rand(n_points_per_class, 2)/2 - 0.25 + (0.25, 0.75))
        labels.append(np.full((n_points_per_class,), 3))

        data = np.vstack(data)
        labels = np.concatenate(labels)

        test_idx = np.random.choice(data.shape[0], test_points, replace=False)
        testData = data[test_idx]
        testLabels = labels[test_idx]
        trainData = np.delete(data, test_idx, axis=0)
        trainLabels = np.delete(labels, test_idx, axis=0)

        print(f"train dataset size: {trainData.shape}, test dataset size: {testData.shape}")

        knn = kNNAlgorithm(metric='minkowski2', voting='idw', weights='mutual_info', method='mat').fit(trainData, trainLabels)

        st = time.time()
        mat_d = knn._computeDistanceMatrix(testData)
        mat_time = time.time() - st

        knn.set_params(method='scipy')
        st = time.time()
        scipy_d = knn._computeDistanceMatrix(testData)
        scipy_time = time.time() - st

        err = np.sqrt(np.sum((scipy_d - mat_d)**2))

        mat_times.append(mat_time)
        scipy_times.append(scipy_time)
        errs.append(err)

        print(f"Has runned with: scipy time={scipy_time:5f}s, mat time={mat_time:5f}s error={err:.6f}")

    plt.figure(figsize=(15, 9))
    plt.subplot(311)
    plt.plot(n_points_trials, mat_times)
    plt.grid('on')
    plt.xlabel('dataset size')
    plt.ylabel('mat_time (s)')

    plt.subplot(312)
    plt.plot(n_points_trials, scipy_times)
    plt.grid('on')
    plt.xlabel('dataset size')
    plt.ylabel('scipy_time (s)')

    plt.subplot(313)
    x = np.linspace(min(errs), max(errs))
    y = stats.norm(loc=np.mean(errs), scale=np.std(errs))
    plt.plot(x, y.pdf(x))
    plt.grid('on')
    plt.xlabel('error')
    plt.ylabel('probability density')

    plt.tight_layout()
    outpath = Path('results/performance')
    outpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath / 'mat_scipy_comp.png', dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    n_data_complexity()
    n_features_complexity()
    scipy_vs_mat()