import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from sklearn.base import ClassifierMixin
from sklearn.neighbors.base import NeighborsBase

class ENN(ClassifierMixin, NeighborsBase):
    
    def __init__(self, k=3):
        self.k = k    
        
    def features(self, X, Y):
        classes = np.unique(Y)
        num_classes = len(classes)
        tree = KDTree(X)
        num_rows = X.shape[0]
        t_origin = np.array([]).reshape(0,self.k)
        dist_map = np.array([]).reshape(0,self.k)
        labels = np.array([]).reshape(0,self.k)

        for row in range(num_rows):
            dist, idx = tree.query(X[row].reshape(1,-1), k = self.k+1)
            dist = dist[0][1:]
            idx = idx[0][1:]
            dist_map = np.append(dist_map, np.array(dist).reshape(1,self.k), axis=0)
            labels = np.append(labels, np.array(Y[idx]).reshape(1,self.k),axis=0)

        for class in classes:
            sum_same_class = np.sum(Y == class)
            same_label = labels[Y.ravel() == class,:]
            same_class = same_label.ravel()
            t_origin = np.append(t_origin, len(same_class[same_class == class]) / (sum_same_class*float(self.k)))

        return dist_map, labels, t_origin, classes, num_classes   
    
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.k_dist = self.features(X, Y)[0]
        self.k_labels = self.features(X, Y)[1]
        self.t_origin = self.features(X, Y)[2]
        self.classes = self.features(X, Y)[3]
        self.num_classes = self.features(X, Y)[4]
        
        self.amt_each_train_class = []
        for idx,class in enumerate(self.classes):
            self.amt_each_train_class.append(len(Y[Y == class]))

        
    def predict(self, X):
        y_pred = []
        
        for x_test in X:
            norm_dist = []
            for row in self.X_train:
                dist = self.distance.euclidean(row, x_test)
                norm_dist.append(dist)

            norm_dist = np.array(norm_dist)
            sort_dist = np.argsort(norm_dist)
            class_top_NN_test = self.Y_train[sort_dist][:self.k]

            sum_NN_each_class = []
            for class in self.classes:
                sum_NN_each_class.append(np.sum(class_top_NN_test == class))

            T_enn = [0] * self.num_classes
            sum_NN_train = [0] * self.num_classes
            sum_same_NN_train = [0] * self.num_classes

            for idx,class in enumerate(self.classes):
                same_class_index = self.Y_train.ravel() == class
                test_dist = norm_dist[same_class_index]
                train_dist = self.k_dist[same_class_index][:,self.k-1]
                train_class = self.k_labels[same_class_index][:,self.k-1]
                dif_dist = test_dist - train_dist

                delta_dist_0 = dif_dist <= 0
                sum_NN_train[idx] = np.sum(delta_dist_0)

                if sum_NN_train[idx] > 0:
                    sum_same_NN_train[idx] = np.sum(train_class[delta_dist_0] == class)

            for j in range(self.num_classes):
                dif_NN_classes = sum_NN_train[j] - sum_same_NN_train[j]
                same_class_amt_ratio = np.array(sum_same_NN_train) / (np.array(self.amt_each_train_class)*float(self.k))
                same_class_sum_ratio = np.sum(same_class_amt_ratio) - sum_same_NN_train[j]/(self.amt_each_train_class[j]*float(self.k))                    

                T_enn[j] = (dif_NN_classes + sum_NN_each_class[j] - self.t_origin[j] * self.k) / ((self.amt_each_train_class[j]+1)*self.k) - same_class_sum_ratio    

            y_pred.append(self.classes[np.argmax(T_enn)])

        return y_pred    