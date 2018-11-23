# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:46:54 2018

@author: Adn
"""

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.models import load_model
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

def predict():

    model = load_model('my_model.h5')

    train = pd.read_csv('csvTrain.csv')
    
    labels = train.iloc[:,0].values.astype('int32')
    X_train = (train.iloc[:,1:].values).astype('float32')
    test = pd.read_csv('testData.csv')
    X_test = (test.iloc[:,1:].values).astype('float32')
    
    data = X_test[[-1],:]
    y_train = np_utils.to_categorical(labels)

    scale = np.max(X_train)
    scale2 = np.max(X_test)

    X_train /= scale
    X_test /= scale
    #X_test /= scale

    input_dimens = X_train.shape[1]
    nb_classes = y_train.shape[1]
    
    prediksi = model.predict_classes(X_test[[-1],:])

    print(len(X_test))
    print(data)
    print("prediksi : ",prediksi)
    
    return prediksi
    
    