from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.datasets import imdb
from keras import backend as K




import numpy as np
import sys
import time


import urllib.request
import os
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils



#     H2O imports

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch

def data_mnist_feed():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()


	X_train = np.squeeze(X_train.reshape((X_train.shape[0], -1)))
	X_test = np.squeeze(X_test.reshape((X_test.shape[0], -1)))

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	#nb_classes = len(np.unique(y_train))
	#y_train = np_utils.to_categorical(y_train, nb_classes)
	#y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)



def data_fashion_feed():
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

	X_train = np.squeeze(X_train.reshape((X_train.shape[0], -1)))
	X_test = np.squeeze(X_test.reshape((X_test.shape[0], -1)))
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	#nb_classes = len(np.unique(y_train))
	#y_train = np_utils.to_categorical(y_train, nb_classes)
	#y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)

def data_imdb():
	num_words=5000
	skip_top= 20
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words,skip_top =skip_top)

	X_train = np.zeros((len(x_train),num_words))
	X_test= np.zeros((len(x_test),num_words))

	for i in range(1,len(x_train)):
		X_train[i,np.unique(x_train[i])[1:]] = True
	for i in range(1,len(x_train)):
		X_test[i,np.unique(x_test[i])[1:]] = True

	#nb_classes = len(np.unique(y_train))
	#y_train = np_utils.to_categorical(y_train, nb_classes)
	#y_test = np_utils.to_categorical(y_test, nb_classes)		

	return (X_train, y_train), (X_test, y_test) 



def data_letters():
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

	#nb_classes = len(np.unique(y_train))

	y_train = [ord(char.lower()) - 97 for char in y_train]
	y_test = [ord(char.lower()) - 97 for char in y_test]

	#y_train = np_utils.to_categorical(y_train, nb_classes)
	#y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)




def load_datasets(name):
	if name=="mnist":
		return data_mnist_feed()
	if name=="cifar10":
		sys.exit("Cifar consume demasiada memoria. No se permite su uso")
	if name=="fashion":
		return data_fashion_feed()
	if name=="imdb":
		return data_imdb()
	if name=="letters":
		return data_letters()

	sys.exit("Wrong dataset name.")



if __name__=='__main__':
	if len(sys.argv)==4:
		datasets = sys.argv[1]
		mode = sys.argv[2]

		#Log info only
		arquitecture= sys.argv[3]
	else:
		sys.exit("Error in command line. No dataset supplied")


	TIME_LIMIT = 12*60*60
	TIME_LIMIT = 1*2*60



	print("Starting H2O")
	h2o.init()

	print("Leyendo ",datasets)

	(X_train, y_train), (X_test, y_test) = load_datasets(datasets)

	data = np.c_[X_train,y_train]
	test = np.c_[X_test,y_test]

	data_h2o = h2o.H2OFrame(data)
	test_h2o = h2o.H2OFrame(test)

	x = data_h2o.columns
	y = data_h2o.columns[len(data_h2o.columns)-1]
	x.remove(y)


	data_h2o[y] = data_h2o[y].asfactor()
	test_h2o[y] = test_h2o[y].asfactor()

	ss=data_h2o.split_frame(seed=1)
	train_h2o=ss[0]
	valid_h2o=ss[1]






	if mode == "Authomatic":
		print("Executing ", datasets, "with ", mode, " mode.\n")

		start_time = time.time()

		aml = H2OAutoML(max_runtime_secs=TIME_LIMIT, seed=1, include_algos = ["DeepLearning"], verbosity = 'info')
		aml.train(x=x, y=y, training_frame=train_h2o,validation_frame=valid_h2o)

				
		print("--- %s seconds ---" % (time.time() - start_time))


		print("Evalutation of best performing model:")
		lb = aml.leaderboard
		lb.head(rows=lb.nrows)

		print(aml.leader)
		print(aml.leaderboard)

		preds =aml.leader.predict(test_h2o)

		#Calculate final ACC

		res = preds == test_h2o[y]
		acc = np.sum(res.as_data_frame().iloc[:,0])/len(res)

		print("Final acc, selected model: ", acc*100)
		print("Time consumed: ",(time.time() - start_time)/3600," hours")


	else:
		if mode == "Guided":
			print("Executing ", datasets, "with ", mode, " mode.\n")

			start_time = time.time()


			DL_params = {	'rate': [i * 0.01 for i in range(1, 11)],
							'activation': ['TanhWithDropout','RectifierWithDropout'],
							'hidden_dropout_ratios': [0.1,0.25,0.5],
							'hidden': [256,128,64,32]
			}
			search_criteria = {'strategy': 'RandomDiscrete', 'max_runtime_secs':TIME_LIMIT,'seed': 1}

			DL_random_grid = H2OGridSearch(	model = H2ODeepLearningEstimator,
											grid_id ='DL_random_grid',
											hyper_params=DL_params,
											search_criteria=search_criteria)

			DL_random_grid.train(x=x,y=y,
								training_frame=data_h2o,
								validation_frame=valid_h2o)

			DL_random_grid_v1 = DL_random_grid.get_grid(decreasing=True)
			best_DL_model = DL_random_grid_v1.models[0]

			preds =  best_DL_model.predict(test_h2o) 

			print(DL_random_grid_v1.models)
			print(DL_random_grid)


			print("--- %s seconds ---" % (time.time() - start_time))


			print("Evalutation of best performing model:")

			#Calculate final ACC
			res = preds == test_h2o[y]
			acc = np.sum(res.as_data_frame().iloc[:,0])/len(res)

			print("Final acc, selected model: ", acc*100)
			print("Time consumed: ",(time.time() - start_time)/3600," hours")





		else:
			print("Error. ", mode, "is not defined. Expected either Convolutional or Feedforward")


