import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import matplotlib.pyplot as plt
from lstm_ts_viz_class import TSViz

from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from tkinter import *
import sys
import webbrowser
#The Datasets
import random
dat = [random.randint(0,100) for i in range(200)]
random_data=[]
for i in range(1, len(dat)):
	random_data.append(dat[i] - dat[i - 1])

#airline_data = [112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0, 118.0, 115.0, 126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0, 145.0, 150.0, 178.0, 163.0, 172.0, 178.0, 199.0, 199.0, 184.0, 162.0, 146.0, 166.0, 171.0, 180.0, 193.0, 181.0, 183.0, 218.0, 230.0, 242.0, 209.0, 191.0, 172.0, 194.0, 196.0, 196.0, 236.0, 235.0, 229.0, 243.0, 264.0, 272.0, 237.0, 211.0, 180.0, 201.0, 204.0, 188.0, 235.0, 227.0, 234.0, 264.0, 302.0, 293.0, 259.0, 229.0, 203.0, 229.0, 242.0, 233.0, 267.0, 269.0, 270.0, 315.0, 364.0, 347.0, 312.0, 274.0, 237.0, 278.0, 284.0, 277.0, 317.0, 313.0, 318.0, 374.0, 413.0, 405.0, 355.0, 306.0, 271.0, 306.0, 315.0, 301.0, 356.0, 348.0, 355.0, 422.0, 465.0, 467.0, 404.0, 347.0, 305.0, 336.0, 340.0, 318.0, 362.0, 348.0, 363.0, 435.0, 491.0, 505.0, 404.0, 359.0, 310.0, 337.0, 360.0, 342.0, 406.0, 396.0, 420.0, 472.0, 548.0, 559.0, 463.0, 407.0, 362.0, 405.0, 417.0, 391.0, 419.0, 461.0, 472.0, 535.0, 622.0, 606.0, 508.0, 461.0, 390.0, 432.0, 450.0]

growth_data = [6.0, 14.0, -3.0, -8.0, 14.0, 13.0, 0.0, -12.0, -17.0, -15.0, 14.0, -3.0, 11.0, 15.0, -6.0, -10.0, 24.0, 21.0, 0.0, -12.0, -25.0, -19.0, 26.0, 5.0, 5.0, 28.0, -15.0, 9.0, 6.0, 21.0, 0.0, -15.0, -22.0, -16.0, 20.0, 5.0, 9.0, 13.0, -12.0, 2.0, 35.0, 12.0, 12.0, -33.0, -18.0, -19.0, 22.0, 2.0, 0.0, 40.0, -1.0, -6.0, 14.0, 21.0, 8.0, -35.0, -26.0, -31.0, 21.0, 3.0, -16.0, 47.0, -8.0, 7.0, 30.0, 38.0, -9.0, -34.0, -30.0, -26.0, 26.0, 13.0, -9.0, 34.0, 2.0, 1.0, 45.0, 49.0, -17.0, -35.0, -38.0, -37.0, 41.0, 6.0, -7.0, 40.0, -4.0, 5.0, 56.0, 39.0, -8.0, -50.0, -49.0, -35.0, 35.0, 9.0, -14.0, 55.0, -8.0, 7.0, 67.0, 43.0, 2.0, -63.0, -57.0, -42.0, 31.0, 4.0, -22.0, 44.0, -14.0, 15.0, 72.0, 56.0, 14.0, -101.0, -45.0, -49.0, 27.0, 23.0, -18.0, 64.0, -10.0, 24.0, 52.0, 76.0, 11.0, -96.0, -56.0, -45.0, 43.0, 12.0, -26.0, 28.0, 42.0, 11.0, 63.0, 87.0, -16.0, -98.0, -47.0, -71.0, 42.0, 18.0]
fall_data = [-18.0, -42.0, 71.0, 47.0, 98.0, 16.0, -87.0, -63.0, -11.0, -42.0, -28.0, 26.0, -12.0, -43.0, 45.0, 56.0, 96.0, -11.0, -76.0, -52.0, -24.0, 10.0, -64.0, 18.0, -23.0, -27.0, 49.0, 45.0, 101.0, -14.0, -56.0, -72.0, -15.0, 14.0, -44.0, 22.0, -4.0, -31.0, 42.0, 57.0, 63.0, -2.0, -43.0, -67.0, -7.0, 8.0, -55.0, 14.0, -9.0, -35.0, 35.0, 49.0, 50.0, 8.0, -39.0, -56.0, -5.0, 4.0, -40.0, 7.0, -6.0, -41.0, 37.0, 38.0, 35.0, 17.0, -49.0, -45.0, -1.0, -2.0, -34.0, 9.0, -13.0, -26.0, 26.0, 30.0, 34.0, 9.0, -38.0, -30.0, -7.0, 8.0, -47.0, 16.0, -3.0, -21.0, 31.0, 26.0, 35.0, -8.0, -21.0, -14.0, 6.0, 1.0, -40.0, 0.0, -2.0, -22.0, 19.0, 18.0, 33.0, -12.0, -12.0, -35.0, -2.0, 12.0, -13.0, -9.0, -5.0, -20.0, 16.0, 22.0, 15.0, 0.0, -21.0, -6.0, -9.0, 15.0, -28.0, -5.0, -5.0, -26.0, 19.0, 25.0, 12.0, 0.0, -21.0, -24.0, 10.0, 6.0, -15.0, -11.0, 3.0, -14.0, 15.0, 17.0, 12.0, 0.0, -13.0, -14.0, 8.0, 3.0, -14.0, -6.0]
epoc=100
mse=[]
vmse=[]
i=0
play=0

def callback(event):
    webbrowser.open_new(r"http://www.github.com/amitrajitbose/lstm-time-series-viz")

def startScreen():
	i=0
	fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
	fig.add_subplot(111).plot(0)
	fig.suptitle('Evaluation')
	canvas =FigureCanvasTkAgg(fig, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row=1,column=6,columnspan=1)	
	fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
	fig.add_subplot(111).plot(0)
	fig.suptitle('Training Set')
	canvas =FigureCanvasTkAgg(fig, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row=2,column=1,columnspan=5)
	fig = Figure(figsize=(4, 2), dpi=100,frameon=False)
	fig.add_subplot(111).plot(0)
	fig.suptitle('Testing Set')
	canvas = FigureCanvasTkAgg(fig, master=master)
	canvas.draw()
	canvas.get_tk_widget().grid(row=2,column=6,columnspan=1)
	Label(master, text=str('Epoch='+str(i))).grid(row=1, column=4, sticky=S)
	link = Label(master, text='About', fg='blue', cursor='hand2')
	link.grid(row=3, column=1, sticky=W, padx=0)
	link.bind("<Button-1>", callback)
	

def playIt():
	if(var1.get()=='Select Dataset'):
		messagebox.showinfo("Error", "Select A Series To Plot")
		return
	elif(var1.get()=='Sine Curve'):
		data = np.sin(np.linspace(-5*np.pi, 5*np.pi, 201))
	elif(var1.get()=='Cosine Curve'):
		data = np.cos(np.linspace(-5*np.pi, 5*np.pi, 201))
	elif(var1.get()=='Increasing Sales'):
		data = growth_data[:]
	elif(var1.get()=='Decreasing Sales'):
		data = fall_data[:]
	elif(var1.get()=='Random Data'):
		data = random_data[:]
	play=1
	try:
		epoc=max(0,int(w3.get()))
	except:
		messagebox.showinfo("Error", "Enter A Positive Value For Epoch")
		return
	mse=[]
	vmse=[]
	maxMse=0
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
		maxMse=max(maxMse,max(*mse,*vmse))
		plt.xlim(0,epoc)
		ax1=fig.add_subplot(111,xlim=(0,epoc),ylim=(0,maxMse+(maxMse/10))).plot(mse)
		ax2=fig.add_subplot(111,xlim=(0,epoc),ylim=(0,maxMse+(maxMse/10))).plot(vmse)
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
		Label(master, text=str('Epoch='+str(i))).grid(row=1, column=4, sticky=S) 
		i=i+1
		master.update()
		#time.sleep(0.0001)

master = Tk() 
master.title('Visualizer') 
startScreen()

Label(master,text='Dropout\n(x 0.01)').grid(row=1,column=1,sticky=N)
w1 = Scale(master, from_=0, to=90)
w1.grid(row=1,column=1,sticky=W) 

Label(master,text='Lag\n').grid(row=1,column=2,sticky=N)
w2 = Scale(master, from_=2, to=10)
w2.grid(row=1,column=2,sticky=W, pady=30) 

Label(master,text='Max Epoch').grid(row=1,column=4,pady=50,sticky=N)
w3 = Entry(master,width=10)
w3.grid(row=1,column=4,pady=30,sticky=N)

Label(master,text='Test Ratio\n(in %)').grid(row=1,column=3,sticky=N)
w4 = Scale(master, from_=10, to=50)
w4.grid(row=1,column=3,sticky=W)


lst1 = ['Select Dataset','Sine Curve','Cosine Curve','Random Data', 'Increasing Sales', 'Decreasing Sales']
var1 = StringVar()
drop = OptionMenu(master,var1,*lst1)
var1.set('Select Dataset') #default
drop.grid(row=1, column=4, pady=30)
B = Button(text ="Start", command = playIt)
B.grid(row=1,column=5,padx=40, pady=50,sticky=N)

rst = Button(text ="Reset", command = startScreen)
rst.grid(row=1, column=5,padx=40,sticky=S, pady=65)

ex = Button(text ="Quit", command = sys.exit)
ex.grid(row=1, column=5,sticky=S,padx=40, pady=10)
#plt.plot(a)
#Label(master, text=var1).grid(row=1, sticky=W) 

mainloop() 