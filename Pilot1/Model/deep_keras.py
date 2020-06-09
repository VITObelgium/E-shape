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
import glob
Test_nr = r'Test4'
if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr)): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr))
os.chdir(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr))
dir_data = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper'
iterations = 100
df_scores = []
fields_wrong_classif = []
fields_good_classif = []
for p in range(iterations):
#df_validation=pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\df_validation_coh_VHVV_fAPAR_update1.0.csv")
#df_calibration=pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\calibration\df_calibration_coh_VHVV_fAPAR_update1.0.csv")
    df_calibration = pd.read_csv(glob.glob(os.path.join(dir_data,'calibration','{}','df_calibration_*iteration{}.csv').format(Test_nr,str(p)))[0])
    df_validation = pd.read_csv(glob.glob(os.path.join(dir_data,'validation','{}','df_validation_*iteration{}.csv').format(Test_nr,str(p)))[0])


    x_train = df_calibration.iloc[0:df_calibration.shape[0],1:df_calibration.shape[1]-1]
    y_train = pd.DataFrame(df_calibration.iloc[0:df_calibration.shape[0],df_calibration.shape[1]-1])
    weight_train=np.ones(y_train.shape)
    weight_train[y_train==1]=10

    x_test = df_validation.iloc[0:df_validation.shape[0],1:df_validation.shape[1]-1]
    y_test = pd.DataFrame(df_validation.iloc[0:df_validation.shape[0],df_validation.shape[1]-1])
    weight_test=np.ones(y_test.shape)
    weight_test[y_test==1]=10
    if p == 32:
        print('h')

    #getting rid of nan values in coherence data
    x_train=x_train.fillna(method='ffill')
    x_test=x_test.fillna(method='ffill')
    #np.sum(np.isnan(x_train))

    #### building the model

    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #input layer ### Dense means the layer type. Dense: all nodes from the previous layer connect to the nodes in the current layer
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu')) #input layer # activation takes into account non-linear relationships
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) ### output layer, only one node namely the prediction #sigmoid put the range of the output between 0 and 1

    ### compiling the model

    model.compile(loss='binary_crossentropy', # way to calculate difference between predicted and observed. Binary crossenentropy useful when training binary classifier
                  optimizer=keras.optimizers.Adam(lr=0.001), #optimize defines the learning rate (adam is a good optimizer) #lr: learning rate
                  metrics=['accuracy'])

    logger = CSVLogger('fit_update1.0_iteration{}.log'.format(str(p)), separator=',', append=False)
    early_stopping=EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False) # early stopping will stop model before epoch reached if the model stops improving. Patience after how many epochs with no improvement stop with running
    checkpoint=ModelCheckpoint('model_update1.0_iteration{}.h5'.format(str(p)), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq = 'epoch') #Save the weight of the model so that they can be used later

    model.fit(x_train, y_train,sample_weight=weight_train.squeeze(), # fit function is used to train the model
              epochs=100, # the number of cycles the model goes through the data
              batch_size=16, validation_data=(x_test,y_test,weight_test.squeeze()),callbacks=[logger,early_stopping,checkpoint])


    logdata=pd.read_csv('fit_update1.0_iteration{}.log'.format(str(p)))

    score = model.evaluate(x_test, y_test, batch_size=16)
    df_scores.append(pd.concat([pd.DataFrame([score[0]], index = ['iteration_{}'.format(str(p))], columns = (['loss'])), pd.DataFrame([score[1]], index = ['iteration_{}'.format(str(p))], columns= (['accuracy']))], axis= 1))
    print(score)


    # plt.plot(logdata['loss'])
    # plt.plot(logdata['val_loss'])
    # plt.show()

    loaded_model = load_model('model_update1.0_iteration{}.h5'.format(str(p)))

    predictions = loaded_model.predict(x_test)
    th=0.5
    predictions[predictions>=th]=1
    predictions[predictions<th]=0
    df_validation['predictions']=predictions
    fields_wrong_classif.extend(df_validation[df_validation['y'] != df_validation['predictions']].ID_field.to_list())
    fields_good_classif.extend(df_validation[df_validation['y'] == df_validation['predictions']].ID_field.to_list())
    if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}'.format(Test_nr)): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}'.format(Test_nr))
    df_validation.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\df_observed_predict_update1.0_iteration{}.csv'.format(Test_nr,str(p)), index = False)
    #pdb.set_trace()

df_scores = pd.concat(df_scores)
df_scores['Best_Score'] = 1-df_scores['loss']+ df_scores['accuracy']
df_scores.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\df_scores_iterations.csv'.format(Test_nr))


###  discriminate between fields of which harvest wrongly detected and non-harvest

### harvest detection issue
fields_harvest_problems = [item for item in fields_wrong_classif if not 'before' in item]
fields_harvest_problems = [item for item in fields_harvest_problems if not 'after' in item]
fields_harvest_problems = pd.DataFrame(fields_harvest_problems, columns = ['harvest_not_detected'])
df_harvest_fields_issues = pd.DataFrame(fields_harvest_problems['harvest_not_detected'].value_counts())
df_harvest_fields_issues.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\df_fields_harvest_detection_problem_counts.csv'.format(Test_nr))

#### non-harvest detection issue
fields_non_harvest_before_problems = [item for item in fields_wrong_classif if 'before' in item]
fields_non_harvest_before_problems = [item.split('_before')[0] for item in fields_non_harvest_before_problems]
fields_non_harvest_before_problems = pd.DataFrame(fields_non_harvest_before_problems, columns= ['non_harvest_before_not_detected'])
df_non_harvest_before_issues = pd.DataFrame(fields_non_harvest_before_problems['non_harvest_before_not_detected'].value_counts())
df_non_harvest_before_issues.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\df_fields_non_harvest_before_detection_problem_counts.csv'.format(Test_nr))

fields_non_harvest_after_problems =  [item for item in fields_wrong_classif if 'after' in item]
fields_non_harvest_after_problems =  [item.split('_after')[0] for item in fields_non_harvest_after_problems]
fields_non_harvest_after_problems = pd.DataFrame(fields_non_harvest_after_problems, columns= ['non_harvest_after_not_detected'])
df_non_harvest_after_issues = pd.DataFrame(fields_non_harvest_after_problems['non_harvest_after_not_detected'].value_counts())
df_non_harvest_after_issues.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\df_fields_non_harvest_after_detection_problem_counts.csv'.format(Test_nr))

