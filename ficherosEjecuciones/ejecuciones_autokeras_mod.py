from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.datasets import imdb

from keras import backend as K


from autokeras import ImageClassifier
from autokeras import MlpModule
from autokeras.backend.torch.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.preprocessor import OneHotEncoder
from autokeras.backend.torch import DataTransformerMlp


import numpy as np
import sys
import time



import urllib.request
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




def load_datasets(name):
	if name=="mnist":
		return mnist.load_data()
	if name=="cifar10":
		sys.exit("Cifar consume demasiada memoria. No se permite su uso")
	if name=="fashion":
		return fashion_mnist.load_data()
	if name=="imdb":
		return imdb.load_data()
	if name=="letters":
		return	load_letters()


	sys.exit("Wrong dataset name.")


def transform_y(y_train):
    # Transform y_train.
    y_encoder = OneHotEncoder()
    y_encoder.fit(y_train)
    y_train = y_encoder.transform(y_train)
    return y_train, y_encoder



if __name__=='__main__':

	if len(sys.argv)==5:
		datasets = sys.argv[1]
		mode = sys.argv[2]

		#Log info only
		arquitecture= sys.argv[3]
		modified=sys.argv[4]
	else:
		sys.exit("Error in command line. No dataset supplied")



	time_limit = 12*60*60

	#Lectura de datos
	print("Leyendo ",datasets)
	(x_train, y_train), (x_test, y_test) =load_datasets(datasets)

	#Creacion de los folds ------ No hay folds por ahora

	#Reshape

	if mode=='Convolutional':
		print("Executing ", datasets, "with ", mode, " arquitecture.\n")
		x_train = x_train.reshape(x_train.shape+(1,))
		x_test = x_test.reshape(x_test.shape+(1,))


		start_time = time.time()
		clf = ImageClassifier(verbose=True,augment= False)

		clf.fit(x_train,y_train, time_limit = time_limit, max_iter_num=8)


		print("--- %s seconds ---" % (time.time() - start_time))
		#Save the model
		clf.export_autokeras_model(datasets+"_"+mode+"_"+arquitecture+"_"+modified)


		#clf.final_fit(x_train, y_train, x_test, y_test, retrain=False)
		y = clf.evaluate(x_test,y_test)
		print("Accuracy final:", y*100)
		print("Time consumed: ",time_limit/3600," hours")
		
	else:
		if mode=="Feedforward":
			print("Executing ", datasets, "with ", mode, " arquitecture.\n")



			x_train = np.squeeze(x_train.reshape((x_train.shape[0], -1)))
			x_test = np.squeeze(x_test.reshape((x_test.shape[0], -1)))
			y_train, y_encoder = transform_y(y_train)
			y_test, _ = transform_y(y_test)

			start_time = time.time()
			mlpModule = MlpModule(loss=classification_loss, metric=Accuracy, searcher_args={}, verbose=True)
			# specify the fit args
			data_transformer = DataTransformerMlp(x_train)
			train_data = data_transformer.transform_train(x_train, y_train)
			test_data = data_transformer.transform_test(x_test, y_test)
			fit_args = {
				"n_output_node": y_encoder.n_classes,
				"input_shape": x_train.shape,
				"train_data": train_data,
				"test_data": test_data
				}
			mlpModule.fit(n_output_node=fit_args.get("n_output_node"),
				input_shape=fit_args.get("input_shape"),
				train_data=fit_args.get("train_data"),
				test_data=fit_args.get("test_data"),
				time_limit=time_limit)


			#mlpModule.export_autokeras_model(datasets+"_"+mode+"_"+arquitecture+"_"+modified)
			y = mlpModule.evaluate(x_test,y_test)
			print("Accuracy final:", y*100)
			print("Time consumed: ",time_limit/3600," hours")



		else:
			print("Error. ", mode, "is not defined. Expected either Convolutional or Feedforward")
	K.clear_session()


