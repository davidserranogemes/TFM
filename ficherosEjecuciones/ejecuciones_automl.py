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

training_parameters = {
	'epochs': 1000,
	'batch_size': 8
}

ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

ml_predictor.train(raw_training_data = df_train,model_names=['DeepLearningClassifier'], training_params = training_parameters,  ml_for_analytics = False)

score = ml_predictor.score(df_test, df_test['0'])

print("Accuracy final:", score)
print("Time consumed: ",time_limit/3600," hours")


#def train(self, raw_training_data, user_input_func=None, optimize_final_model=None, write_gs_param_results_to_file=True, perform_feature_selection=None, verbose=True, X_test=None, y_test=None, ml_for_analytics=True, take_log_of_y=None, model_names=None, perform_feature_scaling=None, calibrate_final_model=False, _scorer=None, scoring=None, verify_features=False, training_params=None, grid_search_params=None, compare_all_models=False, cv=2, feature_learning=False, fl_data=None, optimize_feature_learning=False, train_uncertainty_model=False, uncertainty_data=None, uncertainty_delta=None, uncertainty_delta_units=None, calibrate_uncertainty=False, uncertainty_calibration_settings=None, uncertainty_calibration_data=None, uncertainty_delta_direction=None, advanced_analytics=None, analytics_config=None, prediction_intervals=None, predict_intervals=None, ensemble_config=None, trained_transformation_pipeline=None, transformed_X=None, transformed_y=None, return_transformation_pipeline=False, X_test_already_transformed=False, skip_feature_responses=None, prediction_interval_params=None):