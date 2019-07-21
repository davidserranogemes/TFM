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


#Libraries required by Hyperas

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform



#Hyperas need that you define de model with the boundaries where Hyperas search
def create_model_feedforward(X_train,y_train,X_test,y_test):
	num_epoch=1*500
	nb_classes = y_train.shape[1]

	model = Sequential()
	model.add(Dense(512, input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Dropout({{uniform(0, 1)}}))
	model.add(Dense({{choice([256, 512, 1024])}}))
	model.add(Activation({{choice(['relu', 'sigmoid'])}}))
	model.add(Dropout({{uniform(0, 1)}}))

	# If we choose 'four', add an additional fourth layer
	if {{choice(['three', 'four'])}} == 'four':
		model.add(Dense(100))

		# We can also choose between complete sets of layers

		model.add({{choice([Dropout(0.5), Activation('linear')])}})
		model.add(Activation('relu'))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
	 					 optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

	result = model.fit(X_train, y_train,
				batch_size={{choice([64, 128])}},
				epochs=num_epoch,
				verbose=2,
				validation_split=0.1)
	#get the highest validation accuracy of the training epochs
	validation_acc = np.amax(result.history['val_acc']) 
	print('Best validation acc of epoch:', validation_acc)
	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def create_model_convolutional(X_train,y_train,X_test,y_test):
	nb_classes = y_train.shape[1]

	num_epoch=1*500

	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',
	                        input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout({{uniform(0, 0.5)}}))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout({{uniform(0, 0.5)}}))

	if {{choice(['two','three'])}} == 'three':
		model.add(Convolution2D(128, 3, 3, border_mode='same'))
		model.add(Activation('relu'))
		model.add(Convolution2D(128, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout({{uniform(0, 0.5)}}))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	 # let's train the model using SGD + momentum (how original).
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	
	model.compile(loss='categorical_crossentropy',
	              optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
	              metrics=['accuracy'])

	result = model.fit(X_train, y_train,
	          batch_size={{choice([64, 128])}},
	          nb_epoch=num_epoch,
	          verbose=2,
	          validation_split=0.1)
	validation_acc = np.amax(result.history['val_acc']) 
	print('Best validation acc of epoch:', validation_acc)
	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}



def data_mnist():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = X_train.reshape(X_train.shape+(1,))
	X_test = X_test.reshape(X_test.shape+(1,))

	nb_classes = len(np.unique(y_train))
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)


def data_mnist_feed():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()


	X_train = np.squeeze(X_train.reshape((X_train.shape[0], -1)))
	X_test = np.squeeze(X_test.reshape((X_test.shape[0], -1)))
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	nb_classes = len(np.unique(y_train))
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)


def data_fashion():
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
	
	X_train = X_train.reshape(X_train.shape+(1,))
	X_test = X_test.reshape(X_test.shape+(1,))
	nb_classes = len(np.unique(y_train))

	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)
	

def data_fashion_feed():
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

	X_train = np.squeeze(X_train.reshape((X_train.shape[0], -1)))
	X_test = np.squeeze(X_test.reshape((X_test.shape[0], -1)))
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	nb_classes = len(np.unique(y_train))
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)

def data_imdb():
	num_words=10000
	skip_top= 20
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words,skip_top =skip_top)

	X_train = np.zeros((len(x_train),num_words))
	X_test= np.zeros((len(x_test),num_words))

	for i in range(1,len(x_train)):
		X_train[i,np.unique(x_train[i])[1:]] = True
	for i in range(1,len(x_train)):
		X_test[i,np.unique(x_test[i])[1:]] = True

	nb_classes = len(np.unique(y_train))
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)		

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

	nb_classes = len(np.unique(y_train))

	y_train = [ord(char.lower()) - 97 for char in y_train]
	y_test = [ord(char.lower()) - 97 for char in y_test]

	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)





if __name__=='__main__':
	if len(sys.argv)==4:
		datasets = sys.argv[1]
		mode = sys.argv[2]

		#Log info only
		arquitecture= sys.argv[3]
	else:
		sys.exit("Error in command line. No dataset supplied")


	print("Leyendo ",datasets)
	
	MAX_EVALS = 20


	if mode == "Convolutional":
		print("Executing ", datasets, "with ", mode, " arquitecture.\n")
	
		start_time = time.time()

		if datasets=="mnist":
			best_run, best_model = optim.minimize(model=create_model_convolutional,
                                      data=data_mnist,
                                      algo=tpe.suggest,
                                      max_evals=MAX_EVALS,
                                      trials=Trials())
			print("--- %s seconds ---" % (time.time() - start_time))

			(X_train, Y_train), (X_test, Y_test) = data_mnist()

		if datasets=="fashion":
			best_run, best_model = optim.minimize(model=create_model_convolutional,
                                      data=data_fashion,
                                      algo=tpe.suggest,
                                      max_evals=MAX_EVALS,
                                      trials=Trials())
			print("--- %s seconds ---" % (time.time() - start_time))

			(X_train, Y_train), (X_test, Y_test) = data_fashion()
		
		print("Evalutation of best performing model:")
		print(best_model.evaluate(X_test, Y_test))

		#Calculate final ACC
		print("Time consumed: ",time_limit/3600," hours")


	else:
		if mode == "Feedforward":
			print("Executing ", datasets, "with ", mode, " arquitecture.\n")

			start_time = time.time()

			if datasets=="mnist":
				best_run, best_model = optim.minimize(model=create_model_feedforward,
	                                      data=data_mnist_feed,
	                                      algo=tpe.suggest,
	                                      max_evals=MAX_EVALS,
	                                      trials=Trials())
				
				print("--- %s seconds ---" % (time.time() - start_time))

				(X_train, Y_train), (X_test, Y_test) = data_mnist_feed()


			if datasets=="fashion":
				best_run, best_model = optim.minimize(model=create_model_feedforward,
	                                      data=data_fashion_feed,
	                                      algo=tpe.suggest,
	                                      max_evals=MAX_EVALS,
	                                      trials=Trials())

				print("--- %s seconds ---" % (time.time() - start_time))

				(X_train, Y_train), (X_test, Y_test) = data_fashion_feed()
			
			if datasets=="imdb":
				best_run, best_model = optim.minimize(model=create_model_feedforward,
	                                      data=data_imdb,
	                                      algo=tpe.suggest,
	                                      max_evals=MAX_EVALS,
	                                      trials=Trials())

				print("--- %s seconds ---" % (time.time() - start_time))

				(X_train, Y_train), (X_test, Y_test) = data_imdb()

			if datasets=="letters":
				best_run, best_model = optim.minimize(model=create_model_feedforward,
	                                      data=data_letters,
	                                      algo=tpe.suggest,
	                                      max_evals=MAX_EVALS,
	                                      trials=Trials())

				print("--- %s seconds ---" % (time.time() - start_time))

				(X_train, Y_train), (X_test, Y_test) = data_letters()


			print("Evalutation of best performing model:")
			print(X_train.shape)
			print(X_test.shape)

			print(best_model.evaluate(X_test, Y_test))

			#Calculate final ACC
			print("Time consumed: ",time_limit/3600," hours")





		else:
			print("Error. ", mode, "is not defined. Expected either Convolutional or Feedforward")
	K.clear_session()



