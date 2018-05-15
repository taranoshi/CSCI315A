import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
import csv

data = pd.read_csv('~/wine.csv')
print(data)
data['F1'], data['F2'], data['F3'], data['F4'], data['F5'], data['F6'], data['F7'], data['F8'], data['F9'], data['F10'], data['F11'], data['F12'], data['F13'] = data['features'].str.split(',').str
del data['features']
#data['Id'] = 
data['Target'] = data['label']
del data['label']
data.insert(0, 'Id', range(0, 0 + len(data)))
print(data)
#data.to_csv("dataset.csv")

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
dump_svmlight_file(X,y,'wine_dataset.libsvm',zero_based=False, comment=None, query_id=None, multilabel=False)