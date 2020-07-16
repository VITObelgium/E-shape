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
Test_nr = r'Test12'
if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr)): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr))
output_dir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr)
os.chdir(output_dir)
dir_data = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper'
iterations = 30
df_scores = []
fields_wrong_classif = []
fields_good_classif = []
run_model = True
if run_model:
    for p in range(iterations):
    #df_validation=pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\df_validation_coh_VHVV_fAPAR_update1.0.csv")
    #df_calibration=pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\calibration\df_calibration_coh_VHVV_fAPAR_update1.0.csv")
        df_calibration = pd.read_csv(glob.glob(os.path.join(dir_data,'calibration','{}','df_calibration_*iteration{}.csv').format(Test_nr,str(p)))[0])
        df_validation = pd.read_csv(glob.glob(os.path.join(dir_data,'validation','{}','df_validation_*iteration{}.csv').format(Test_nr,str(p)))[0])


        x_train = df_calibration.iloc[0:df_calibration.shape[0],1:df_calibration.shape[1]-1]
        y_train = pd.DataFrame(df_calibration.iloc[0:df_calibration.shape[0],df_calibration.shape[1]-1])
        weight_train=np.ones(y_train.shape)
        weight_train[y_train==1]=10

        x_val = df_validation.iloc[0:df_validation.shape[0],1:df_validation.shape[1]-1]
        y_val = pd.DataFrame(df_validation.iloc[0:df_validation.shape[0],df_validation.shape[1]-1])
        weight_val=np.ones(y_val.shape)
        weight_val[y_val==1]=10

        df_test = pd.read_csv(glob.glob(os.path.join(dir_data, 'test', '{}'.format(Test_nr), 'df_test_*.csv'))[0])
        x_test = df_test.iloc[0:df_test.shape[0],1:df_test.shape[1]-1]
        y_test = pd.DataFrame(df_test.iloc[0:df_test.shape[0],df_test.shape[1]-1])
        weight_test = np.ones(y_test.shape)
        weight_test[y_test == 1] = 10

        #getting rid of nan values in coherence data
        x_train=x_train.fillna(method='ffill')
        x_val=x_val.fillna(method='ffill')
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
                      optimizer=keras.optimizers.Adam(lr= 0.0005), #optimize defines the learning rate (adam is a good optimizer) #lr: learning rate   #0.001 : original
                      metrics=['accuracy'])

        logger = CSVLogger('fit_update1.0_iteration{}.log'.format(str(p)), separator=',', append=False)
        early_stopping=EarlyStopping(monitor='val_loss', min_delta=0.01, patience= 10, verbose=1, mode='auto', baseline=None, restore_best_weights=False) # early stopping will stop model before epoch reached if the model stops improving. Patience after how many epochs with no improvement stop with running
        checkpoint=ModelCheckpoint('model_update1.0_iteration{}.h5'.format(str(p)), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq = 'epoch') #Save the weight of the model so that they can be used later

        model.fit(x_train, y_train,sample_weight=weight_train.squeeze(), # fit function is used to train the model
                  epochs=200, # the number of cycles the model goes through the data #100
                  batch_size=32, validation_data=(x_val,y_val,weight_val.squeeze()),callbacks=[logger,early_stopping,checkpoint]) # batch size was  16


        logdata=pd.read_csv('fit_update1.0_iteration{}.log'.format(str(p)))

        score = model.evaluate(x_test, y_test, batch_size=16)
        print(score)


        # plt.plot(logdata['loss'])
        # plt.plot(logdata['val_loss'])
        # plt.show()

        loaded_model = load_model('model_update1.0_iteration{}.h5'.format(str(p)))

        predictions = loaded_model.predict(x_test)
        th=0.5
        harvest_predict_confid_mean= np.nanmean(predictions[np.where(y_test == 1)])
        harvest_predict_confid_med = np.nanmedian(predictions[np.where(y_test == 1)])

        harvest_predict_confid_std = np.std(predictions[np.where(y_test == 1)])

        ####### plotting of performance model
        fig, (ax1) = plt.subplots(1, figsize=(15, 10))
        plt.plot(logdata['epoch'], logdata['loss'], label = 'Training_loss', color = 'red')
        plt.plot(logdata['epoch'], logdata['val_loss'], label = 'Validation_loss', color = 'blue')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc = 'upper right')
        ax1.set_title('Model Test_{}_no_early_stopping'.format(str(p)))
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir,'Model_{}_iteration_{}_no_early_stopping.png'.format(str(Test_nr),str(p))))
        plt.close()

        df_test['predictions_prob']=predictions
        predictions[predictions >= th] = 1
        predictions[predictions < th] = 0
        df_test['predictions'] = predictions
        harvest_false_pos_confid_med = np.nanmedian(df_test.loc[((df_test['y'] == 0) & (df_test['predictions'] == 1))]['predictions_prob'].values)#### see how certain the model is when harvest is detected in a no-harvest period
        harvest_false_pos_confid_mean = np.nanmean(df_test.loc[((df_test['y'] == 0) & (df_test['predictions'] == 1))]['predictions_prob'].values)  #### see how certain the model is when harvest is detected in a no-harvest period
        harvest_false_pos_confid_std = np.nanstd(df_test.loc[((df_test['y'] == 0) & (df_test['predictions'] == 1))]['predictions_prob'].values)
        df_scores.append(pd.concat([pd.DataFrame([score[0]], index=['iteration_{}'.format(str(p))], columns=(['loss'])),
                                pd.DataFrame([score[1]], index=['iteration_{}'.format(str(p))],
                                             columns=(['accuracy'])),
                                pd.DataFrame([harvest_predict_confid_med], index=['iteration_{}'.format(str(p))],
                                             columns=(['Median_confid_harvest'])),
                                    pd.DataFrame([harvest_predict_confid_mean], index=['iteration_{}'.format(str(p))],
                                                 columns=(['Mean_confid_harvest'])),
                                pd.DataFrame([harvest_predict_confid_std], index=['iteration_{}'.format(str(p))],
                                             columns=(['STDV_confid_harvest'])),
                                    pd.DataFrame([harvest_false_pos_confid_med], index=['iteration_{}'.format(str(p))],
                                                 columns=(['Median_confid_true_pos'])),
                                    pd.DataFrame([harvest_false_pos_confid_mean], index=['iteration_{}'.format(str(p))],
                                                 columns=(['Mean_confid_true_pos'])),
                                    pd.DataFrame([harvest_false_pos_confid_std], index=['iteration_{}'.format(str(p))],
                                                 columns=(['STDV_confid_true_pos']))
                                    ], axis=1))
        fields_wrong_classif.extend(df_test[df_test['y'] != df_test['predictions']].ID_field.to_list())
        fields_good_classif.extend(df_test[df_test['y'] == df_test['predictions']].ID_field.to_list())
        if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}'.format(Test_nr)): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}'.format(Test_nr))
        df_test.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\df_observed_predict_update1.0_iteration{}.csv'.format(Test_nr,str(p)), index = False)
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

###### calculate the amount of under or over detections (omission error)
dir_validation_files = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}'.format(Test_nr)
validation_files = glob.glob(os.path.join(dir_validation_files,'df_observed_*iteration*.csv'))
iteration_numbers_files = [item.split('_')[-1:][0].split('.csv')[0] for item in validation_files]
df_harvest_overdetection = []
h = 0
for validation_file in validation_files:
    df = pd.read_csv(validation_file)
    df_cond = pd.DataFrame(df['predictions'] == df['y'])
    df_cond = df_cond.iloc[0:np.where(df['y'] == 1)[0].size]
    incorrectly_reference_sites = len(np.where(df_cond == False)[0])
    tot_reference_sites = np.where(df['y'] == 1)[0].size
    omission_error = (incorrectly_reference_sites/tot_reference_sites) *100
    df_harvest_overdetection.append(pd.concat([pd.DataFrame([omission_error], index=['{}'.format(str(iteration_numbers_files[h]))], columns=(['omission_error'])),
                                pd.DataFrame([incorrectly_reference_sites], index=['{}'.format(str(iteration_numbers_files[h]))],
                                             columns=(['incorrectly_classified']))], axis=1))
    h +=1
df_harvest_overdetection = pd.concat(df_harvest_overdetection, axis= 0)
average_ommission_iterations = df_harvest_overdetection['omission_error'].mean()
tot_wrongly_class_iterations = df_harvest_overdetection['incorrectly_classified'].sum()
df_harvest_overdetection = pd.concat([df_harvest_overdetection, pd.concat([pd.DataFrame([average_ommission_iterations], index=['summary'], columns=(['omission_error'])),
                                pd.DataFrame([tot_wrongly_class_iterations], index=['summary'], columns = (['incorrectly_classified']))], axis= 1)], axis = 0)
df_harvest_overdetection.to_csv(os.path.join(dir_validation_files,'df_harvest_overdetection_{}.csv'.format(Test_nr)))

