#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:29:26 2019

@author: davidserranogemes
"""

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold


dir_path = os.path.dirname(os.path.realpath(__file__))


#def load_complete_mnist(path= '/datasets/mnist.npz_FILES'):
def load_complete_mnist(path= '/datasets/mnist.npz'):
    dir_path = os.path.dirname(os.path.realpath(__file__))    
    file_path = dir_path+path
    
    with np.load(file_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    
    X = np.concatenate([x_train,x_test], axis = 0)
    Y = np.concatenate([y_train,y_test], axis = 0)
    
    return X,Y

def define_k_fold_mnist(X,Y,k=10):
    skf = StratifiedKFold(n_splits=k)
    it_fold = 1
    
    dirname = "mnist_folds/"
    for train_index, test_index in skf.split(X,Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        filename_x_train = dirname + "mnist-" + str(it_fold) + "-fold-train_x"
        filename_y_train = dirname + "mnist-" + str(it_fold) + "-fold-train_y"
        filename_x_test = dirname + "mnist-" + str(it_fold) + "-fold-test_x"
        filename_y_test = dirname + "mnist-" + str(it_fold) + "-fold-test_y"

        np.save(file = filename_x_train,arr = x_train)
        np.save(file = filename_y_train,arr = y_train)
        np.save(file = filename_x_test,arr = x_test)
        np.save(file = filename_y_test,arr = y_test)
        it_fold = it_fold+1
        
        
X,Y=load_complete_mnist()
define_k_fold_mnist(X,Y)