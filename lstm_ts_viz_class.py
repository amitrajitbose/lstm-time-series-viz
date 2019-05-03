from __future__ import absolute_import, division, print_function, unicode_literals
#!pip install -q tensorflow==2.0.0-alpha0
import tensorflow as tf
#print(tf.__version__)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import matplotlib.pyplot as plt
class TSViz(object):
    """LSTM On Time Series Data Visualizer"""
    def __init__(self, data, verbose,epoch,lag,dropout):
        self.data = data
        self.lag = lag
        self.epoch = epoch
        self.test_size = 0.3
        self.dropout = dropout
        self.model = None
        self.shuffle = False
        self.validation_split = 0.2
        self.verbose = verbose
        self.X_train=[]
        self.Y_train=[]
        self.X_test=[]
        self.Y_test=[]

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
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        return X_train,X_test,Y_train,Y_test

    def build_model(self):
        model=Sequential()
        model.add(LSTM(10, input_shape=(self.lag-1,1)))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        self.model = model
        if(self.verbose):
            model.summary()

    def fit(self):
        history = self.model.fit(self.X_train, self.Y_train, epochs=self.epoch, shuffle=self.shuffle, batch_size=4,validation_data=(self.X_test,self.Y_test), verbose=self.verbose)
        return history

    def predict(self, X, Y):
        #plt.plot(Y)
        #plt.plot(self.model.predict(X))
        #plt.legend(['Actual','Predicted'],loc='upper right')
        #plt.show()
        return Y,self.model.predict(X)
     
        
    
    def mastermethod(self):
        X_train,X_test,Y_train,Y_test = self.data_prepare()
        self.build_model()
        self.fit(X_train, Y_train)
        aa,bb=self.predict(self.X_train, self.Y_train)
        plt.plot(aa)
        plt.plot(bb)
        plt.legend(['Actual','Predicted'],loc='upper right')
        plt.show()
        aa,bb=self.predict(self.X_test, self.Y_test)
        plt.plot(aa)
        plt.plot(bb)
        plt.legend(['Actual','Predicted'],loc='upper right')
        plt.show()