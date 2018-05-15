import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import dump_svmlight_file
from sklearn import datasets
from sklearn.datasets import load_boston, load_wine #doesnt work


boston = datasets.load_boston()
X = boston.data
y = boston.target
dump_svmlight_file(X,y,'boston.libsvm',zero_based=False, comment=None, query_id=None, multilabel=False)

X, y = load_wine(return_X_y=True)
dump_svmlight_file(W,y,'wine.libsvm',zero_based=False, comment=None, query_id=None, multilabel=False)