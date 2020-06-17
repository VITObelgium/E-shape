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
import re
import geopandas as gpd

outdir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\6_daily_window_data'
def harvest_window_eval(df):
    harvest_detections_window = len(np.where(df.predictions == 1)[0]) ### amount of harvest events in window according to the model
    df['prediction_date_window'] = df['prediction_date_window'].values.astype('datetime64[D]')
    df['Harvest_date'] = df['Harvest_date'].values.astype('datetime64[D]')
    try:
        difference_harvest_prediction = abs(df.loc[df['predictions'] ==1]['prediction_date_window'].mean()- df['Harvest_date'].mean())
    except:
        difference_harvest_prediction = np.nan
    if harvest_detections_window != 0:
        harvest_detections_window = [1]*df.shape[0]
    else:
        harvest_detections_window = [0] * df.shape[0]
    df['harvest_recognized'] = harvest_detections_window
    df['error_harvest_prediction'] = difference_harvest_prediction
    return df
Window_val = True
window_harvest = 18 ### amount of days from the harvest date to start the window specified from the middle of the period 18-3 => 15 days window around harvest date
Test_nr = r'Test4'
amount_metrics_model = 8
if Window_val:
    if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format('6_daily_window_data')): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format('6_daily_window_data'))
os.chdir(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr))
dir_data = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper'
df_validation = pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\6_daily_window_data\df_harvest_model_6daily.csv")
iterations = 100
fields_no_harvest_detection = []
average_harvest_prediction_error = []
for p in range(iterations):
    print('ITERATION: {}'.format(str(p)))
    x_test = df_validation.iloc[0:df_validation.shape[0], 1:amount_metrics_model+1]
    loaded_model = load_model('model_update1.0_iteration{}.h5'.format(str(p)))
    predictions = loaded_model.predict(x_test)
    th = 0.5
    predictions[predictions >= th] = 1
    predictions[predictions < th] = 0
    df_validation['predictions'] = predictions
    df_validation['Diff_harvest_2'] = pd.to_timedelta(df_validation['Diff_harvest_2'])
    df_validation['Diff_harvest_3'] = pd.to_timedelta(df_validation['Diff_harvest_3'])
    ##### check if harvest is detected in a certain window
    df_validation_sample_harvest_period = df_validation[(df_validation['Diff_harvest_2'] >= pd.Timedelta('-{} days'.format(str(window_harvest)))) & (df_validation['Diff_harvest_3'] <= pd.Timedelta('{} days'.format(str(window_harvest)))) ]
    df_validation_sample_harvest_period.reset_index(drop = True, inplace= True)
    ids_fields = df_validation_sample_harvest_period.ID_field.to_list()
    ids_fields = [item.rsplit('_',1)[0]for item in ids_fields] #only split at last _ of the string
    ids_fields = pd.DataFrame(ids_fields, columns=['ID_field_grouped'])
    df_validation_sample_harvest_period = pd.concat([df_validation_sample_harvest_period,ids_fields], axis = 1)
    ids_fields = list(set(ids_fields))
    df_validation_sample_harvest_period = df_validation_sample_harvest_period.groupby(['ID_field_grouped']).apply(harvest_window_eval) ### function to determine of harvest will detectec in the window per field


    ##### make some statistics showing the goodness of the harvest model to predict the date within the specified window
    error_harvest_fields = df_validation_sample_harvest_period[df_validation_sample_harvest_period['harvest_recognized'] == 0].ID_field_grouped.to_list()
    #average_harvest_prediction_error_field = df_validation_sample_harvest_period.loc[df_validation_sample_harvest_period['ID_field_grouped'].isin(shp_TAP.id.to_list())]['error_harvest_prediction'].dropna().values.mean()
    average_harvest_prediction_error_field = df_validation_sample_harvest_period['error_harvest_prediction'].dropna().values.mean()

    try:
        average_harvest_prediction_error_field = average_harvest_prediction_error_field.days # daily resolution note that 1 days 23 hours becomes 1 => maybe to good
    except:
        average_harvest_prediction_error_field = 0
    fields_no_harvest_detection.extend(list(set(error_harvest_fields)))
    average_harvest_prediction_error.append(pd.DataFrame([average_harvest_prediction_error_field], columns= ['average_prediction_error_harvest'], index = ['model_{}'.format(str(p))]))
average_harvest_prediction_error = pd.concat(average_harvest_prediction_error, axis = 0)
fields_no_harvest_detection = pd.DataFrame(fields_no_harvest_detection, columns=['harvest_not_detected'])
df_harvest_fields_issues = pd.DataFrame(fields_no_harvest_detection['harvest_not_detected'].value_counts())

df_harvest_fields_issues.to_csv(os.path.join(outdir,'df_harvest_fields_issues.csv'))
average_harvest_prediction_error.to_csv(os.path.join(outdir,'df_average_harvest_prediction_error_TAP.csv'))


#### some small tests for TAP fields
df_harvest_fields_issues.index.name = 'ID_fields'
df_harvest_fields_issues = df_harvest_fields_issues.reset_index()
shp_TAP  = pd.read_csv(r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.csv")
df_harvest_fields_issues = df_harvest_fields_issues.loc[df_harvest_fields_issues.ID_fields.isin(shp_TAP.id.to_list())]





