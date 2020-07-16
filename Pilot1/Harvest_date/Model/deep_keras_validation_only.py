import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import pandas as pd
import pickle 
import pdb
import csv
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import os

os.chdir(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Mehrdad')
df_validation=pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Mehrdad\validation\df_validation.csv")
df_validation = df_validation.drop(columns = ['t59_coh1','t59_coh2','t59_coh3','t59_coh4'])
#df_validation = pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Mehrdad\validation\Validation_data_eshape_2019_update_order.csv")
df_calibration=pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Mehrdad\calibration\df_calibration_ro59_removed.csv")
#df_validation = df_validation.drop(columns = ['ro88_coh_1','ro88_coh_2','ro88_coh_3','ro88_coh_4'])
x_train = df_calibration.iloc[0:470,0:11]
y_train = df_calibration.iloc[0:470,11:12] #### the collum we want to predict (target)
weight_train=np.ones(y_train.shape)
weight_train[y_train==1]=10

x_test = df_validation.iloc[:,0:11]
y_test = df_validation.iloc[:,11:12]
weight_test=np.ones(y_test.shape)
weight_test[y_test==1]=10
#
# #getting rid of nan values in coherence data
x_train=x_train.fillna(method='ffill')
x_test=x_test.fillna(method='ffill')

np.sum(np.isnan(x_train))

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #input layer ### Dense means the layer type. Dense: all nodes from the previous layer connect to the nodes in the current layer
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu')) #input layer # activation takes into account non-linear relationships
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid')) ### output layer, only one node namely the prediction #sigmoid put the range of the output between 0 and 1

model.compile(loss='binary_crossentropy', # way to calculate difference between predicted and observed. Binary crossenentropy useful when training binary classifier
              optimizer=keras.optimizers.Adam(lr=0.001), #optimize defines the learning rate (adam is a good optimizer) #lr: learning rate
              metrics=['accuracy'])

logger = CSVLogger('fit_ro59_removed.log', separator=',', append=False)
early_stopping=EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False) # early stopping will stop model before epoch reached if the model stops improving. Patience after how many epochs with no improvement stop with running
checkpoint=ModelCheckpoint('model_ro59_removed.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1) #Save the weight of the model so that they can be used later

model.fit(x_train, y_train,sample_weight=weight_train.squeeze(), # fit function is used to train the model
          epochs=100, # the number of cycles the model goes through the data
          batch_size=16, validation_data=(x_test,y_test,weight_test.squeeze()),callbacks=[logger,early_stopping,checkpoint])


model.fit(x_train, y_train,sample_weight=weight_train.squeeze(),
          epochs=100,
          batch_size=16, validation_data=(x_test,y_test,weight_test.squeeze()))


logdata=pd.read_csv('fit_ro59_removed.log')

score = model.evaluate(x_test, y_test, batch_size=16)
print(score)


plt.plot(logdata['loss'])
plt.plot(logdata['val_loss'])
plt.show()

loaded_model = load_model('model_ro59_removed.h5')

predictions = loaded_model.predict(x_test)
th=0.5
predictions[predictions>=th]=1
predictions[predictions<th]=0
df_validation['predictions']=predictions
#df_validation.to_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Mehrdad\validation\Validation_data_eshape_2019_predictions_ro59_removed.csv")
pdb.set_trace()

