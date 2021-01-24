#############################################################
#############################################################
#############################################################


import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def plot_SVC(X_train, y_train, X_test, y_pred, svc):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='x', s=30, cmap=plt.cm.Paired)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='o', s=30, cmap=plt.cm.Paired)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')


if __name__ == "__main__":

    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def run_svm_dataset1():
        X1, y1, X2, y2 = generate_data_set1()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        # Write here your SVM code and choose a linear kernel
        svc = SVC(kernel='linear').fit(X_train, y_train)
        pred = svc.predict(X_test)
        correct = np.sum(pred == y_test)

        # plot the graph with the support_vectors_
        plot_SVC(X_train, y_train, X_test, pred, svc)
        plt.title('linear kernel C=1.0')
        plt.show()

        # print on the console the number of correct predictions and the total of predictions
        print(f"function run_svm_dataset1, linear kernel: correctly classified {correct}/{len(y_test)} instances")


    def run_svm_dataset2():
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and choose a linear kernel with the best C parameter
        svc = SVC(kernel='rbf').fit(X_train, y_train)
        pred = svc.predict(X_test)
        correct = np.sum(pred == y_test)

        # plot the graph with the support_vectors_
        plot_SVC(X_train, y_train, X_test, pred, svc)
        plt.title(f'gaussian kernel')
        plt.show()

        # print on the console the number of correct predictions and the total of predictions
        print(f"function run_svm_dataset2, rbf kernel: correctly classified {correct}/{len(y_test)} instances")



    def run_svm_dataset3(C=1.0):
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and use a gaussian kernel 
        svc = SVC(kernel='linear', C=C).fit(X_train, y_train)
        pred = svc.predict(X_test)
        correct = np.sum(pred == y_test)

        # plot the graph with the support_vectors_
        plot_SVC(X_train, y_train, X_test, pred, svc)
        plt.title(f'linear kernel C={C}')
        plt.show()

        # print on the console the number of correct predictions and the total of predictions
        print(f"function run_svm_dataset3, C={C}, linear kernel: correctly classified {correct}/{len(y_test)} instances")
        return correct/len(y_test)



#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS      
    run_svm_dataset1()   # data distribution 1
    run_svm_dataset2()   # data distribution 2
    for c in np.linspace(0.1, 0.5, 5):
        run_svm_dataset3(C=c)

#############################################################
#############################################################
#############################################################
