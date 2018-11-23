# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:13:48 2018

@author: Adn
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 27 18:04:43 2018

@author: Adn
"""
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

def train(th, ep):
    
    train = pd.read_csv('csvTrain.csv')
    labels = train.iloc[:,0].values.astype('int32')
    X_train = (train.iloc[:,1:].values).astype('float32')
    test = pd.read_csv('testData.csv')
    X_test = (test.iloc[:,1:].values).astype('float32')

    
    y_train = np_utils.to_categorical(labels)
   
    
    errTh = th
    epos = ep
    
    #normalisasi
    scale = np.max(X_train)
    scale2 = np.max(X_test)

    X_train /= scale
    X_test /= scale
    #X_test /= scale


    input_dimens = X_train.shape[1]
    nb_classes = y_train.shape[1]

    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []
        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('val_acc'))
            if logs.get('loss') < errTh:
                model.stop_training = True
            return logs.get('loss')
            
    cb = AccuracyHistory()
    # Here's a  MLP 
    model = Sequential()
    model.add(Dense( 6, activation = 'relu', input_shape=(input_dimens,)))
    model.add(Dense( 8, activation = 'relu'))
    model.add(Dense( 8, activation = 'relu'))
    model.add(Dense( 6, activation = 'softmax'))

    model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])

    print("Training...")
    history = model.fit(X_train, y_train, nb_epoch=epos, batch_size=5, validation_split=0.1, verbose=1,  callbacks=[cb])

    plt.figure(figsize = [8, 6])
    plt.plot(history.history['loss'], 'r', linewidth = 3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth = 3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize = 18)
    plt.xlabel('Epochs', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)

    plt.figure(figsize = [8, 6])
    plt.plot(history.history['acc'], 'r', linewidth= 3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth = 3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    print("Generating test predictions...")
    preds = model.predict_classes(X_test, verbose=0)


    print("prediksi 4: ",model.predict_classes(X_train[[30],:]))
    print("prediksi 5: ",model.predict_classes(X_train[[45],:]))
    print("prediksi 1: ",model.predict_classes(X_train[[5],:]))
    print("prediksi 2: ",model.predict_classes(X_train[[16],:]))
    print("prediksi : ",model.predict_classes(X_test[[-1],:]))

    
    model.save('my_model.h5')
    print("epos = ", len(history.epoch))
    print(history.history['loss'])
    print("Data :", data)
    return str(history.history['acc'][-1]), str(history.history['loss'][-1]), str(len(history.epoch))
#print("prediksi 3: ",model.predict_classes(X_test))
#print("prediksi real",model.predict_classes(X_test[[0],:]))