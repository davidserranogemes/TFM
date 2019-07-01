from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model



import numpy as np
import pandas as pd

import sys
import time

def load_datasets(name):
	if name=="mnist":
		return mnist.load_data()
	if name=="cifar10":
		return cifar10.load_data()
	if name=="fashion":
		return fashion_mnist.load_data()
	sys.exit("Wrong dataset name.")





if __name__=='__main__':

	if len(sys.argv)==3:
		datasets = sys.argv[1]
		mode = sys.argv[2]
	else:
		if len(sys.argv)==2:
			datasets = sys.argv[1]
			mode = 'Convolutional'
		else:
			sys.exit("Error in command line. No dataset supplied")



	time_limit = 1*60*60

	#Lectura de datos
	print("Leyendo ",datasets)
	(x_train, y_train), (x_test, y_test) =load_datasets(datasets)
	x_train = np.squeeze(x_train.reshape((x_train.shape[0], -1)))
	x_test = np.squeeze(x_test.reshape((x_test.shape[0], -1)))

	train = np.column_stack((y_train, x_train))
	test = np.column_stack((y_test, x_test))

	print(x_train.shape)
	print(x_test.shape)

	df_train = pd.DataFrame(train, columns= list(map(str,(list(range(0,train.shape[1]))))))
	df_test = pd.DataFrame(test, columns= list(map(str,(list(range(0,test.shape[1]))))))

	column_descriptions = {
	    '0': 'output',
	}

	ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

	ml_predictor.train(df_train,model_names=['DeepLearningClassifier'])

	score = ml_predictor.score(df_test, df_test.0)

	print("Accuracy final:", score)
	print("Time consumed: ",time_limit/3600," hours")


