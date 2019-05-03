import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import time
import matplotlib.pyplot as plt
from tkinter import *
master = Tk() 
master.title('LSTM') 

#The Data
data = 5*np.random.randn(1,80).reshape(80)

#Preprocess
lag=3
series_matrix=np.array([[j for j in data[i:i+lag]] for i in range(0,len(data)-lag+1)])
X,Y=series_matrix[:,0:-1],series_matrix[:,[2]]
test_size=int(0.3*X.shape[0])
train_size=X.shape[0]-test_size
X_train,X_test,Y_train,Y_test=X[0:train_size,:],X[0:test_size,:],Y[0:train_size],Y[0:test_size]
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#The model
model=Sequential()
model.add(LSTM(10, input_shape=(2,1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

#Epoch
epoc=500
mse=[]
vmse=[]
while (epoc>0):
	history=model.fit(X_train, Y_train, validation_split=0.2, epochs=1, shuffle=False,verbose=0)
#Evals
	fig = Figure(figsize=(5, 3), dpi=100,frameon=False)
	mse.append(*history.history['loss'])
	vmse.append(*history.history['val_loss'])
	fig.add_subplot(111).plot(mse)
	fig.add_subplot(111).plot(vmse)
	#fig.add_subplot(111).plot(history.history['mean_squared_error'])
	fig.legend(['Train','Val'])
	fig.suptitle('Evaluation')
	#fig.xlim(0, 50)
	#fig.ylim(0, 50)
	canvas =FigureCanvasTkAgg(fig, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row=1,column=2)	
#Train Graph
	fig = Figure(figsize=(5, 3), dpi=100,frameon=False)
	fig.add_subplot(111).plot(model.predict(X_train))
	fig.add_subplot(111).plot(Y_train)
	fig.legend(['Predicted','Actual'])
	fig.suptitle('Training Set')
	#fig.xlim(0, 50)
	#fig.ylim(0, 50)
	canvas =FigureCanvasTkAgg(fig, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row=2,column=1)

#Test Graph
	fig = Figure(figsize=(5, 3), dpi=100,frameon=False)
	fig.add_subplot(111).plot(model.predict(X_test))
	fig.add_subplot(111).plot(Y_test)
	fig.legend(['Predicted','Actual'])
	fig.suptitle('Testing Set')
	#fig.xlim(0, 50)
	#fig.ylim(0, 50)
	canvas =FigureCanvasTkAgg(fig, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row=2,column=2)
	Label(master, text=500-epoc).grid(row=1, sticky=W) 
	epoc=epoc-1
	master.update()
	time.sleep(0.1)
#plt.plot(a)
#Label(master, text=var1).grid(row=1, sticky=W) 

mainloop() 