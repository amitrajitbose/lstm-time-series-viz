import matplotlib.pyplot as plt
from lstm_ts_viz_class import TSViz
import numpy as np
data = np.sin(np.linspace(-5*np.pi, 5*np.pi, 201))

tt=TSViz(data,verbose=0,epoch=20,dropout=0.2)
tt.data_prepare()
tt.build_model()
hist=tt.fit()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Train Loss','Val Loss'],loc='upper right')
plt.show()
aa,bb=tt.predict(tt.X_train, tt.Y_train)
plt.plot(aa)
plt.plot(bb)
plt.legend(['Actual','Predicted'],loc='upper right')
plt.show()
aa,bb=tt.predict(tt.X_test, tt.Y_test)
plt.plot(aa)
plt.plot(bb)
plt.legend(['Actual','Predicted'],loc='upper right')
plt.show()