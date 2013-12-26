import json

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

KNN=True
NB=False

def load_iris_data() :

    iris = datasets.load_iris()
   
    return (iris.data, iris.target, iris.target_names)

def knn(X_train, y_train, k_neighbors = 3):

    clf = kNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf

def nb(X_train, y_train):

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)
    
    return clf
