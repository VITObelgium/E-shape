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
Test_nr = r'Test10'
outdir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\6_daily_window_data'.format(Test_nr)
ro_s = ['ro110','ro161']
output_dir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}\6_daily_window_data'.format(Test_nr)
outdir_hist = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\Test10\6_daily_window_data\Histogram_error_harvest_pred'
Test_harvest_detection_accuracies = False
Test_harvest_date_prediction_accuracy = True
histogram_prediction_error = False
Harvest_date_prediction_accuracies = []
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(outdir): os.makedirs(outdir)
def harvest_window_eval(df):
    harvest_detections_window = len(np.where(df.predictions == 1)[0]) ### amount of harvest events in window according to the model
    df['prediction_date_window'] = df['prediction_date_window'].values.astype('datetime64[D]')
    df['Harvest_date'] = df['Harvest_date'].values.astype('datetime64[D]')
    try:
        difference_harvest_prediction = abs(df.loc[df['predictions'] ==1]['prediction_date_window'].mean()- df['Harvest_date'].mean())
        harvest_date_prediction_model = df.loc[df['predictions'] == 1]['prediction_date_window'].mean().dayofyear # the doy of the harvest prediction per field
    except:
        difference_harvest_prediction = np.nan
        harvest_date_prediction_model = np.nan
    if harvest_detections_window != 0:
        harvest_detections_window = [1]*df.shape[0]
    else:
        harvest_detections_window = [0] * df.shape[0]
    df['harvest_recognized'] = harvest_detections_window # columns to indicate if one of the windows for that field an harvest event was detected
    df['error_harvest_prediction'] = difference_harvest_prediction
    df['DOY_harvest_prediction'] = harvest_date_prediction_model
    return df
def combine_orbits_harvest_date(df):
    df_reduc = df.drop_duplicates(subset = ['ID_field_grouped'])
    if df_reduc.shape[0] == 2:
        mean_DOY_harvest_pred_orbits = df_reduc.DOY_harvest_prediction.mean()
        df_reduc['DOY_harvest'] = pd.to_datetime(df_reduc.Harvest_date.values).dayofyear
        harvest_prediction_error =  (df_reduc.DOY_harvest.mean()) - mean_DOY_harvest_pred_orbits
        df['DOY_harvest_prediction'] = [mean_DOY_harvest_pred_orbits]*df.shape[0]
        df['DOY_harvest_error'] = [harvest_prediction_error] * df.shape[0]
        df['DOY_harvest'] = pd.to_datetime(df.Harvest_date.values).dayofyear

    return df
Window_val = True
window_harvest = 60 ### amount of days from the harvest date to start the window specified from the middle of the period 18-3 => 15 days window around harvest date
amount_metrics_model = 10
if Window_val:
    if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}\{}'.format(Test_nr,'6_daily_window_data')): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}\{}'.format(Test_nr,'6_daily_window_data'))
os.chdir(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\{}'.format(Test_nr))
dir_data = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper'
df_validation = pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\6_daily_window_data\df_harvest_model_6daily_TAP_only_ro110_ro161_30_daily_window.csv")
iterations = 30
fields_no_harvest_detection = []
average_harvest_prediction_error = []
thresholds_harvest_date_detection = np.arange(0.5,0.6,0.02)
thresholds_harvest_date_detection = np.array([0.8])
for thr in thresholds_harvest_date_detection:
    print('\nVALIDATION OF THR {}'.format(str(thr)))
    for p in range(iterations):
        print('ITERATION: {}'.format(str(p)))
        x_test = df_validation.iloc[0:df_validation.shape[0], 1:amount_metrics_model+1]
        loaded_model = load_model('model_update1.0_iteration{}.h5'.format(str(p)))
        predictions = loaded_model.predict(x_test)
        df_validation['predictions_prob'] = predictions
        #th = 0.5
        predictions[predictions >= thr] = 1
        predictions[predictions < thr] = 0
        df_validation['predictions'] = predictions
        df_validation['Diff_harvest_2'] = pd.to_timedelta(df_validation['Diff_harvest_2'])
        df_validation['Diff_harvest_3'] = pd.to_timedelta(df_validation['Diff_harvest_3'])

        # ###### Plotting of the probabilities through time:
        # for id in
        # fig, (ax1) = plt.subplots(1, figsize=(15, 10))
        # ax1.scatter(df_validation['prediction_date_window'].values.astype('datetime64[D]'),df_validation['predictions_prob'])
        #

        ##### check if harvest is detected in a certain window
        df_validation_sample_harvest_period = df_validation[(df_validation['Diff_harvest_2'] >= pd.Timedelta('-{} days'.format(str(window_harvest)))) & (df_validation['Diff_harvest_3'] <= pd.Timedelta('{} days'.format(str(window_harvest)))) ]
        df_validation_sample_harvest_period.reset_index(drop = True, inplace= True)
        ids_fields = df_validation_sample_harvest_period.ID_field.to_list()
        ids_fields = [item.rsplit('_',1)[0]for item in ids_fields] #only split at last of the string
        ids_fields = pd.DataFrame(ids_fields, columns=['ID_field_grouped'])
        df_validation_sample_harvest_period = pd.concat([df_validation_sample_harvest_period,ids_fields], axis = 1)
        ids_fields = list(set([item.rsplit('_',1)[0] for item in df_validation_sample_harvest_period['ID_field_grouped']]))
        df_validation_sample_harvest_period = df_validation_sample_harvest_period.groupby(['ID_field_grouped']).apply(harvest_window_eval) ### function to determine of harvest will be detecteced in the window per field

        # ###### Plotting of the probabilities through time per field with the two ro's on top of each other:
        if Test_harvest_detection_accuracies:
            for id in ids_fields:
                fig, (ax1) = plt.subplots(1, figsize=(15, 10))
                colors = ['red','green']
                for ro in ro_s:
                    df_validation_id = df_validation_sample_harvest_period.loc[df_validation_sample_harvest_period['ID_field_grouped'] == id+'_{}'.format(ro)]
                    ax1.scatter(df_validation_id['prediction_date_window'],df_validation_id['predictions_prob'], color = colors[ro_s.index(ro)], label = ro)
                ax1.axvline(x=df_validation_id['Harvest_date'].values[0], color='black', label='Harvest')
                ax1.set_ylabel('Probability_harvest')
                ax1.set_xlabel('Middle date of window')
                ax1.legend(loc='upper right')
                ax1.set_title('Harvest_prediction_prob_time_series_model_{}'.format(str(p)))
                plt.tight_layout()
                if not os.path.exists(os.path.join(output_dir, 'iteration_{}'.format(str(p)))): os.makedirs(os.path.join(output_dir,'iteration_{}'.format(str(p))))
                fig.savefig(os.path.join(output_dir,'iteration_{}'.format(str(p)),'Harvest_prob_time_series_model_{}.png'.format(str(id))))
                plt.close()


        def RMSE(df, ro_s,p,thr):
            field_names = df.ID_field_grouped.unique()
            RMSE_output_df = []
            days_harvest_prediction_error = dict()
            for orbit in ro_s:
                field_names_orbit = [item for item in field_names if orbit in item]
                df_orbit = df_validation_sample_harvest_period.loc[df.ID_field_grouped.isin(field_names_orbit)]
                df_orbit_reduced = df_orbit.drop_duplicates(subset='ID_field_grouped').reset_index(drop=True) # keep only one row per field to limit data reduncy
                df_orbit_reduced['DOY_harvest'] = pd.to_datetime(df_orbit_reduced.Harvest_date.values).dayofyear
                days_harvest_prediction_error.update({'{}'.format(orbit): (df_orbit_reduced.DOY_harvest - df_orbit_reduced.DOY_harvest_prediction)})
                RMSE_harvest_date_prediction = np.sqrt(((df_orbit_reduced.DOY_harvest_prediction-df_orbit_reduced.DOY_harvest)**2).mean())
                RMSE_output_df.append(RMSE_harvest_date_prediction)


            field_names_orbit_combined = [item.rsplit('_', 1)[0] for item in df.ID_field_grouped.to_list()]
            df['ID_field_orbit_comb'] = field_names_orbit_combined
            df = df.groupby(['ID_field_orbit_comb']).apply(combine_orbits_harvest_date)
            df.drop_duplicates(subset = ['ID_field_orbit_comb'], inplace = True)
            df.reset_index(drop = True, inplace = True)
            days_harvest_prediction_error.update({'{}'.format('ro_combined'): (df.DOY_harvest_error)})
            RMSE_harvest_date_prediction = np.sqrt(((df.DOY_harvest_prediction - df.DOY_harvest) ** 2).mean())
            RMSE_output_df.append(RMSE_harvest_date_prediction)
            RMSE_df = pd.DataFrame(np.array(RMSE_output_df)[np.newaxis],columns=(['RMSE_{}'.format(ro_s[0]), 'RMSE_{}'.format(ro_s[1]),'RMSE_ro_combined']),index=['Model_{}_thr_{}'.format(str(p), str(thr))])

            return RMSE_df,days_harvest_prediction_error
        ##### make some statistics showing the goodness of the harvest model to predict the date within the specified window
        error_harvest_fields = df_validation_sample_harvest_period[df_validation_sample_harvest_period['harvest_recognized'] == 0].ID_field_grouped.to_list()
        #average_harvest_prediction_error_field = df_validation_sample_harvest_period.loc[df_validation_sample_harvest_period['ID_field_grouped'].isin(shp_TAP.id.to_list())]['error_harvest_prediction'].dropna().values.mean()
        average_harvest_prediction_error_field = df_validation_sample_harvest_period['error_harvest_prediction'].dropna().values.mean()

        ### calculate the RMSE for the different orbist per model
        RMSE_error, days_harvest_prediction_error = RMSE(df_validation_sample_harvest_period,ro_s,p,thr)
        Harvest_date_prediction_accuracies.append(RMSE_error)

        try:
            average_harvest_prediction_error_field = average_harvest_prediction_error_field.days # daily resolution note that 1 days 23 hours becomes 1 => maybe to good
        except:
            average_harvest_prediction_error_field = 0
        fields_no_harvest_detection.extend(list(set(error_harvest_fields)))
        average_harvest_prediction_error.append(pd.DataFrame([average_harvest_prediction_error_field], columns= ['average_prediction_error_harvest'], index = ['model_{}'.format(str(p))]))

        if histogram_prediction_error:
            keys = list(days_harvest_prediction_error.keys())
            fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(15, 10))
            ax1.hist(x = days_harvest_prediction_error.get('{}'.format(keys[0])).values, width = 1, label = '{}'.format(keys[0]), color = 'red', align = 'mid')
            ax2.hist(x = days_harvest_prediction_error.get('{}'.format(keys[1])).values, width = 1, label = '{}'.format(keys[1]), color = 'blue' ,align = 'mid')
            ax3.hist(x = days_harvest_prediction_error.get('{}'.format(keys[2])).values, width = 1, label = '{}'.format(keys[2]), color = 'black',  align = 'mid')
            ax1.legend(loc = 'upper right')
            ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')
            ax1.set_ylabel('# Fields')
            ax1.set_xlabel('Prediction error (days)')
            ax2.set_ylabel('# Fields')
            ax2.set_xlabel('Prediction error (days)')
            ax3.set_ylabel('# Fields')
            ax3.set_xlabel('Prediction error (days)')
            ax1.set_xlim([-30,30])
            ax2.set_xlim([-30,30])
            ax3.set_xlim([-30,30])

            if not os.path.exists(os.path.join(outdir_hist,'TAP_excl')): os.makedirs(os.path.join(outdir_hist,'TAP_excl'))
            fig.savefig(os.path.join(outdir_hist,'TAP_excl','Model_{}_histogram_harvest_pred_error.png'.format(str(p))))
            plt.close()



average_harvest_prediction_error = pd.concat(average_harvest_prediction_error, axis = 0)
fields_no_harvest_detection = pd.DataFrame(fields_no_harvest_detection, columns=['harvest_not_detected'])
df_harvest_fields_issues = pd.DataFrame(fields_no_harvest_detection['harvest_not_detected'].value_counts())
if Test_harvest_detection_accuracies:
    df_harvest_fields_issues.to_csv(os.path.join(outdir,'df_harvest_fields_issues.csv'))
    average_harvest_prediction_error.to_csv(os.path.join(outdir,'df_average_harvest_prediction_error_TAP.csv'))
if Test_harvest_date_prediction_accuracy:
    Harvest_date_prediction_accuracies  = pd.concat(Harvest_date_prediction_accuracies)
    #Harvest_date_prediction_accuracies.to_csv(os.path.join(outdir,'RMSE_ascending_descending_harvest_date_prediction_thresholds_{}_{}_TAP_excl_ro_combined.csv'.format(str(thresholds_harvest_date_detection.min()),str(thresholds_harvest_date_detection.max()))))





###### code to merge the RMSE for different thresholds together
# RMSE_files = glob.glob(os.path.join(outdir,'RMSE_*TAP_excl_ro_combined.csv'))
# df_RMSE_summ = pd.concat(pd.read_csv(f) for f in RMSE_files)
# df_RMSE_summ = df_RMSE_summ.rename(columns = ({'Unnamed: 0': 'Model_name'}))
# df_RMSE_summ.to_csv(os.path.join(outdir,'RMSE_ascending_descending_harvest_date_prediction_thresholds_TAP_excl_ro_combined.csv'), index = False)





#### some small tests for TAP fields
# df_harvest_fields_issues.index.name = 'ID_fields'
# df_harvest_fields_issues = df_harvest_fields_issues.reset_index()
# shp_TAP  = pd.read_csv(r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.csv")
# df_harvest_fields_issues = df_harvest_fields_issues.loc[df_harvest_fields_issues.ID_fields.isin(shp_TAP.id.to_list())]




