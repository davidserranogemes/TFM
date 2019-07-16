import urllib.request
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold



def load_letters():
	seed = 7

	dirname = 'letters.csv'
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
	local_filename, headers = urllib.request.urlretrieve(url, dirname)  

	n_cols= 17
	n_rows=20000

	X = np.genfromtxt(local_filename, delimiter= ',',usecols = range(1,n_cols))
	y = np.genfromtxt(local_filename, delimiter= ',',usecols = 0, dtype=None)


	skf = StratifiedKFold(n_splits=2, random_state = seed)
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

	return (X_train, y_train), (X_test, y_test)



