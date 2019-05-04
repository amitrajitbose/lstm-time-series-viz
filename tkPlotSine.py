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
import random
random_data = [random.randint(0,100) for i in range(200)]
airline_data = [112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0, 118.0, 115.0, 126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0, 145.0, 150.0, 178.0, 163.0, 172.0, 178.0, 199.0, 199.0, 184.0, 162.0, 146.0, 166.0, 171.0, 180.0, 193.0, 181.0, 183.0, 218.0, 230.0, 242.0, 209.0, 191.0, 172.0, 194.0, 196.0, 196.0, 236.0, 235.0, 229.0, 243.0, 264.0, 272.0, 237.0, 211.0, 180.0, 201.0, 204.0, 188.0, 235.0, 227.0, 234.0, 264.0, 302.0, 293.0, 259.0, 229.0, 203.0, 229.0, 242.0, 233.0, 267.0, 269.0, 270.0, 315.0, 364.0, 347.0, 312.0, 274.0, 237.0, 278.0, 284.0, 277.0, 317.0, 313.0, 318.0, 374.0, 413.0, 405.0, 355.0, 306.0, 271.0, 306.0, 315.0, 301.0, 356.0, 348.0, 355.0, 422.0, 465.0, 467.0, 404.0, 347.0, 305.0, 336.0, 340.0, 318.0, 362.0, 348.0, 363.0, 435.0, 491.0, 505.0, 404.0, 359.0, 310.0, 337.0, 360.0, 342.0, 406.0, 396.0, 420.0, 472.0, 548.0, 559.0, 463.0, 407.0, 362.0, 405.0, 417.0, 391.0, 419.0, 461.0, 472.0, 535.0, 622.0, 606.0, 508.0, 461.0, 390.0, 432.0, 450.0]

epoc=100
mse=[]
vmse=[]
i=0
play=0
def playIt():
	if(var1.get()=='Select Data'):
		messagebox.showinfo("Error", "Select A Series To Plot")
		return
	elif(var1.get()=='Sine'):
		data = np.sin(np.linspace(-5*np.pi, 5*np.pi, 201))
	elif(var1.get()=='Cosine'):
		data = np.cos(np.linspace(-5*np.pi, 5*np.pi, 201))
	elif(var1.get()=='Airline'):
		data = airline_data[:]
	elif(var1.get()=='Random'):
		data = random_data[:]
	play=1
	try:
		epoc=max(0,int(w3.get()))
	except:
		messagebox.showinfo("Error", "Enter A Positive Value For Epoch")
		return
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
		canvas.get_tk_widget().grid(row=1,column=6,columnspan=1)	
	#Train Graph
		fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
		fig.add_subplot(111).plot(tt.predict(tt.X_train, tt.Y_train)[0])
		fig.add_subplot(111).plot(tt.predict(tt.X_train, tt.Y_train)[1])
		fig.legend(['Actual','Predicted'])
		fig.suptitle('Training Set')
		canvas =FigureCanvasTkAgg(fig, master=master)
		canvas.draw()
		canvas.get_tk_widget().grid(row=2,column=1,columnspan=5)

	#Test Graph
		fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
		fig.add_subplot(111).plot(tt.predict(tt.X_test, tt.Y_test)[0])
		fig.add_subplot(111).plot(tt.predict(tt.X_test, tt.Y_test)[1])
		fig.legend(['Actual','Predicted'])
		fig.suptitle('Testing Set')
		canvas = FigureCanvasTkAgg(fig, master=master)
		canvas.draw()
		canvas.get_tk_widget().grid(row=2,column=6,columnspan=1)
		Label(master, text=str('Epoch='+str(i))).grid(row=1, column=4, sticky=W) 
		i=i+1
		master.update()
		#time.sleep(0.0001)

master = Tk() 
master.title('LSTM Visualizer') 

Label(master,text='Dropout(x 0.01)').grid(row=1,column=1,sticky=N)
w1 = Scale(master, from_=10, to=90)
w1.grid(row=1,column=1,sticky=W) 

Label(master,text='Lag').grid(row=1,column=2,sticky=N)
w2 = Scale(master, from_=2, to=10)
w2.grid(row=1,column=2,sticky=W, pady=30) 

Label(master,text='Epochs').grid(row=1,column=4,sticky=N)
w3 = Entry(master,width=10)
w3.grid(row=1,column=4,pady=30,sticky=N)

Label(master,text='Test Ratio(in %)').grid(row=1,column=3,sticky=N)
w4 = Scale(master, from_=10, to=50)
w4.grid(row=1,column=3,sticky=W)


lst1 = ['Select Data','Sine','Cosine','Random', 'Airline']
var1 = StringVar()
drop = OptionMenu(master,var1,*lst1)
var1.set('Select Data') #default
drop.grid(row=2, column=4, pady=20)

B = Button(text ="Start", command = playIt)
B.grid(row=1,column=5,padx=40, pady=50)

ex = Button(text ="Quit", command = sys.exit)
ex.grid(row=1, column=5,sticky=S,padx=40, pady=10)
#plt.plot(a)
#Label(master, text=var1).grid(row=1, sticky=W) 

mainloop() 