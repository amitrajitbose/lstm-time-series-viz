from __future__ import absolute_import, division, print_function, unicode_literals
#!pip install -q tensorflow==2.0.0-alpha0
import tensorflow as tf
#tf.__version__

import numpy as np
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import matplotlib.pyplot as plt
from time import sleep
#plt.ion()
#matplotlib.interactive(True)

class TSViz(object):
    """LSTM On Time Series Data Visualizer"""

    def __init__(self, data, verbose, lag=3, epoch=200, test_size=0.3, dropout=0.2, shuffle=False, validation_split=0.2):
        self.data = data
        self.lag = lag
        self.epoch = epoch
        self.test_size = test_size
        self.dropout = dropout
        self.model = None
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.verbose = verbose

    def data_prepare(self):
        series_matrix=np.array([[j for j in self.data[i:i+self.lag]] for i in range(0,len(self.data)-self.lag+1)])
        X, Y = series_matrix[:,0:-1], series_matrix[:,[-1]]
        test_size=int(self.test_size*X.shape[0])
        train_size=X.shape[0]-test_size
        X_train,X_test,Y_train,Y_test=X[0:train_size,:],X[0:test_size,:],Y[0:train_size],Y[0:test_size]
        X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1) #reshaping, due to keras requirement for LSTM input size
        X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1) #reshaping, due to keras requirement for LSTM input size
        if self.verbose:
          print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
        return X_train,X_test,Y_train,Y_test

    def build_model(self):
        model=Sequential()
        model.add(LSTM(10, input_shape=(self.lag-1,1)))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam',metrics=['mse'])
        self.model = model
        if(self.verbose):
            model.summary()

    def fit(self, X_train, Y_train):
        history = self.model.fit(X_train, Y_train, epochs=self.epoch, shuffle=self.shuffle, validation_split=self.validation_split, verbose=self.verbose)
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.legend(['Training Data','Validating Data'],loc='upper right')
        plt.show()

    def predict(self, X, Y):
        #pass
        plt.plot(Y)
        plt.plot(self.model.predict(X))
        plt.legend(['Actual','Predicted'],loc='upper right')
        plt.show()
    
    def mastermethod(self):
        X_train,X_test,Y_train,Y_test = self.data_prepare()
        self.build_model()
        self.fit(X_train, Y_train)
        self.predict(X_train, Y_train)
        self.predict(X_test, Y_test)


import numpy as np
data = np.sin(np.linspace(-5*np.pi, 5*np.pi, 201))
#for ep in range(1,101,50):
tsvz = TSViz(data, lag=2, epoch=200, verbose=0)
tsvz.mastermethod()
