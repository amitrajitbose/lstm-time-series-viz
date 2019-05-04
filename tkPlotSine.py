import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import matplotlib.pyplot as plt
from lstm_ts_viz_class import TSViz

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import time
import matplotlib.pyplot as plt
from tkinter import *
import sys

#The Data
data = np.sin(np.linspace(-5*np.pi, 5*np.pi, 201))

epoc=100
mse=[]
vmse=[]
i=0
play=0
def playIt():
	play=1
	epoc=max(0,w3.get())
	mse=[]
	vmse=[]
	i=0
	tt=TSViz(data,verbose=0,epoch=1,lag=w2.get(),dropout=w1.get()/100, test_size=w4.get()/100)
	#print("Dropout: ",tt.dropout,"%")
	tt.data_prepare()
	tt.build_model()
	while (i<=epoc and play==1):
		hist=tt.fit()
	#Evals
		fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
		mse.extend(hist.history['loss'])
		vmse.extend(hist.history['val_loss'])
		fig.add_subplot(111).plot(mse)
		fig.add_subplot(111).plot(vmse)
		#fig.add_subplot(111).plot(history.history['mean_squared_error'])
		fig.legend(['Train','Val'])
		fig.suptitle('Evaluation')
		canvas =FigureCanvasTkAgg(fig, master=master)
		canvas.draw()
		canvas.get_tk_widget().grid(row=1,column=3,columnspan=1)	
	#Train Graph
		fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
		fig.add_subplot(111).plot(tt.predict(tt.X_train, tt.Y_train)[0])
		fig.add_subplot(111).plot(tt.predict(tt.X_train, tt.Y_train)[1])
		fig.legend(['Actual','Predicted'])
		fig.suptitle('Training Set')
		canvas =FigureCanvasTkAgg(fig, master=master)
		canvas.draw()
		canvas.get_tk_widget().grid(row=2,column=1,columnspan=2)

	#Test Graph
		fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
		fig.add_subplot(111).plot(tt.predict(tt.X_test, tt.Y_test)[0])
		fig.add_subplot(111).plot(tt.predict(tt.X_test, tt.Y_test)[1])
		fig.legend(['Actual','Predicted'])
		fig.suptitle('Testing Set')
		canvas = FigureCanvasTkAgg(fig, master=master)
		canvas.draw()
		canvas.get_tk_widget().grid(row=2,column=3,columnspan=1)
		Label(master, text=str('Epoch='+str(i))).grid(row=1, sticky=W) 
		i=i+1
		master.update()
		time.sleep(0.0001)

master = Tk() 
master.title('LSTM') 

Label(master,text='Dropout(x 0.01)').grid(row=1,column=1,sticky=N)
w1 = Scale(master, from_=10, to=90)
w1.grid(row=1,column=1,sticky=W) 

Label(master,text='Lag').grid(row=1,column=2,sticky=N)
w2 = Scale(master, from_=2, to=10)
w2.grid(row=1,column=2,sticky=W, pady=30) 

Label(master,text='Epochs').grid(row=1,column=3,sticky=N)
w3 = Scale(master, from_=1, to=200)
w3.grid(row=1,column=3,sticky=W)

Label(master,text='Test Ratio(in %)').grid(row=1,column=4,sticky=N)
w4 = Scale(master, from_=10, to=50)
w4.grid(row=1,column=4,sticky=W)

B = Button(text ="Start", command = playIt)
B.grid(row=1,column=5,sticky=W,padx=40, pady=50)

ex = Button(text ="Quit", command = sys.exit)
ex.grid(row=2, column=5,sticky=W,padx=40, pady=10)
#plt.plot(a)
#Label(master, text=var1).grid(row=1, sticky=W) 

mainloop() 