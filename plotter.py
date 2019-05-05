from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
from tensorflow.keras.utils import plot_model

dropout=0.2
model=Sequential()
model.add(LSTM(30, input_shape=(5-1,1),return_sequences= True))
model.add(Dropout(dropout))
model.add(LSTM(30))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

plot_model(model, to_file="model_arch.png", show_shapes=True, show_layer_names=True, rankdir='TB')