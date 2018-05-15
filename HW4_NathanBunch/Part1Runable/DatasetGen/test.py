import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
import csv

data = pd.read_csv('~/dataset_full_csv.csv')
print(data)
data['F1'], data['F2'] = data['features'].str.split(',', 1).str
del data['features']
#data['Id'] = 
data['Target'] = data['label']
del data['label']
data.insert(0, 'Id', range(0, 0 + len(data)))
print(data)
data.to_csv("dataset.csv")

#df = pd.DataFrame()
#df['Id'] = np.arange(10)
#df['F1'] = np.random.rand(10,)
#df['F2'] = np.random.rand(10,)
#df['Target'] = map(lambda x: -1 if x < 0.5 else 1, np.random.rand(10,))

X = data[np.setdiff1d(data.columns,['Id','Target'])]
y = data.Target

#data['label'] = data.rename(index=equi).index
#X = data['label']
#y = data[data.columns.difference(['Id','label'])]

#print(df)

#dump_svmlight_file(X,y,'dataset.libsvm',zero_based=True,multilabel=False)
dump_svmlight_file(X,y,'dataset.libsvm')