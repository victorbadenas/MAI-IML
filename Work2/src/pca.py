import numpy as np


class PCA:
    """
    PCA Algorithm

    This object is responsible of performing the PCA algorithm in a set of
    data and store the necessary information to retrieve the data and infer
    in the future.

    Parameters:
        n_components : int, default=3
            The number of principal components that the return data must have.

        verbose : bool, default=False
            boolean flag to print intermediate steps and required values 
            to display.

        print_ : bool, default=True
            alias for verbose (in future implementations may be different)

    """
    def __init__(self, n_components=3, *, verbose=False, print_=True):
        self.nComponents = n_components
        self.verbose = verbose
        self.print_ = print_ or verbose
        self.__reset()

    def fit(self, trainData):
        self.__reset()
        trainData = trainData.copy()
        trainData = self.__normalize(trainData)
        eigenValues, eigenVectors = self.__computeEigenVecEigenVal(trainData)
        self.eigenVectors = eigenVectors[:, :self.nComponents]
        self.eigenValues = eigenValues[:self.nComponents]
        return self

    def sortValuesVectors(self, eigenValues, eigenVectors):
        idxs = np.argsort(eigenValues)[::-1]
        eigenValues = eigenValues[idxs]
        eigenVectors = eigenVectors[:, idxs]
        return eigenValues, eigenVectors

    def __computeEigenVecEigenVal(self, data):
        covarianceMatrix = np.cov(data.T)
        if self.print_:
            print(covarianceMatrix)
        eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
        eigenValues, eigenVectors = self.sortValuesVectors(eigenValues, eigenVectors)
        if self.print_:
            self.__displayEigenValuesAndEigenvectors(eigenVectors, eigenValues)
        return eigenValues, eigenVectors

    def __normalize(self, data):
        self.mean_ = data.mean(axis=0, keepdims=True)
        data -= self.mean_
        return data

    def predict(self, data):
        assert self.eigenVectors is not None, "PCA has not been fitted. run pca.PCA.fit(X) first"
        assert self.eigenValues is not None, "PCA has not been fitted. run pca.PCA.fit(X) first"
        assert self.mean_ is not None, "PCA has not been fitted. run pca.PCA.fit(X) first"
        return self.__predict(data)

    def fit_transform(self, trainData):
        self.__reset()
        self.fit(trainData)
        return self.predict(trainData)

    def inverse_transform(self, data):
        assert self.eigenVectors is not None, "PCA has not been fitted. run pca.PCA.fit(X) first"
        assert self.eigenValues is not None, "PCA has not been fitted. run pca.PCA.fit(X) first"
        assert self.mean_ is not None, "PCA has not been fitted. run pca.PCA.fit(X) first"
        return self.__revert(data)

    @staticmethod
    def __displayEigenValuesAndEigenvectors(eigenVectors, eigenValues):
        msg = "Eigenvector {0}: {1} with eigenValue {2}"
        for i in range(len(eigenValues)):
            print(msg.format(i, eigenVectors[:, i], eigenValues[i]))

    def __reset(self):
        self.eigenVectors = None
        self.eigenValues = None
        self.mean_ = None

    def __predict(self, data):
        return (data - self.mean_)@self.eigenVectors

    def __revert(self, data):
        return (data@self.eigenVectors.T) + self.mean_


if __name__ == "__main__":
    trainData = np.random.randn(20, 10)
    n_components = 2

    pca = PCA(n_components=n_components)

    reducedData = pca.fit_transform(trainData)
    assert reducedData.shape[1] == n_components

    reconstructedData = pca.inverse_transform(reducedData)
    assert reconstructedData.shape == trainData.shape

    print(pca.eigenVectors@pca.eigenVectors.T)  # should be a unit matrix

    print(np.mean(reconstructedData - trainData))  # should be close to 0
