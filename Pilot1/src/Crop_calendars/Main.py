import openeo
from openeo.rest.conversions import timeseries_json_to_pandas
import pandas as pd
from openeo.rest.job import RESTJob
import geopandas as gpd
import shapely
from shapely.ops import transform
from functools import partial
import pyproj
import geojson
import numpy as np
import os
import scipy.signal
import json
import requests
import time
import logging
import zipfile
import uuid
import matplotlib.pyplot as plt
from shapely.geometry import  Polygon
from Pilot1.src.Crop_calendars.Main_functions import concat_df_from_Openeo
from Pilot1.src.Crop_calendars.Main_functions import fAPAR_CropSAR_concat_OpenEO
from Pilot1.src.Crop_calendars.Main_functions import fAPAR_CropSAR_concat_API
from Pilot1.src.Crop_calendars.Main_functions import S1_VHVV_ratio_concat
from Pilot1.src.Crop_calendars.Main_functions import moving_window_metrics_extraction
from Pilot1.src.Crop_calendars.Main_functions import combine_cropcalendar_data
from Pilot1.src.Crop_calendars.Main_functions import coherence_concat
from Pilot1.src.Crop_calendars.Main_functions import Plot_time_series_metrics_crop_calendar
from Pilot1.src.Crop_calendars.Main_functions import apply_NN_model
from Pilot1.src.Crop_calendars.Main_functions import Plot_time_series_metrics_crop_calendar_probability
from Pilot1.src.Crop_calendars.Main_functions import RMSE_plotting_against_prob_threshold
from Pilot1.src.Crop_calendars.Main_functions import validate_crop_calendar_event_date
from Pilot1.src.Crop_calendars.Main_functions import find_optimal_model_threshold


####### OPENEO EXTRACTION OF THE REQUIRED METRICS + PRE-PROCESSING OF THE INPUT DATA SO THE MODEL CAN USE IT AND PLOT THE FINAL OUTPUT
# constant parameters
VH_VV_range = [-13,-3.5] #-30, -8 => old wrong dB calculation
fAPAR_range =  [0,1]
metrics = [r'VHVV', 'CropSAR'] #, 'coherence'
crop_calendar_events = ['Harvest_da']
Basefolder = r'S:\eshape\Pilot 1\results\{}\plots'.format(crop_calendar_events[0])
trained_model_dir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10'
outdir_prob_plotting = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10\6_daily_window_data'
iterations = 30
#### dictionaries containing some info on the used datasets to allow crop calendar generation
datasets_dict = {'2018_CAC_sbe': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2018_sbe\32TQQ_2018_CAC_sbe.shp"}

#'2017_CAC_sbe': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2017_sbe\32TQQ_2017_CAC_sbe.shp",
#'2018_CAC_soy': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2018_soy\32TQQ_2018_CAC_soy.shp",
#'2019_CAC_sbe': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2019_sbe\32TQQ_2019_CAC_sbe.shp",
#'2019_CAC_soy': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2019_soy\32TQQ_2019_CAC_soy.shp",
# '2018_Greece': r"S:\eshape\Pilot 1\data\Parcels_greece\35TLF_2018_parcel_ids_greece.shp",



# '2018_Flax': r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp",
#                  '2018_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp",
#                  '2019_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_fields_planting_dates.shp",

#datasets_dict = {'2019_TAP': r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.shp"}
country_dataset_dict = {'2018_CAC_sbe': r"Italy"}
# '2017_CAC_sbe': r"Italy",
#'2018_CAC_soy': r"Italy",
#'2019_CAC_sbe': r"Italy",
#'2019_CAC_soy': r"Italy"
# '2018_Greece': r"Greece",





# '2018_Flax': r"Belgium",
#                  '2018_WIG': r"Belgium",
#                  '2019_WIG': r"Belgium",

#country_dataset_dict = {'2019_TAP': r'Belgium'}
ROs_dataset_dict = {'2018_CAC_sbe': ['ro44', 'ro117', 'ro95']}

# '2017_CAC_sbe': ['ro117', 'ro95'],
#'2018_CAC_soy': ['ro117', 'ro95'],
#'2019_CAC_sbe': ['ro44', 'ro117', 'ro95', 'ro168'],
#'2019_CAC_soy': ['ro44', 'ro117', 'ro95', 'ro168']
# '2018_Greece': ['ro29','ro131', 'ro109'],


# '2018_Flax':['ro110','ro161'],
#                  '2018_WIG': ['ro110','ro161'],
#                  '2019_WIG': ['ro110','ro161'],

#ROs_dataset_dict = {'2019_TAP': ['ro110', 'ro161']}
dict_cropcalendars_data_locations = {'2018_CAC_sbe': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2018_sbe\CAC_2018_sbe_cropcalendars.xlsx"}
# '2017_CAC_sbe': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2017_sbe\CAC_2017_sbe_cropcalendars.xlsx",
#'2018_CAC_soy': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2018_soy\CAC_2018_soy_cropcalendars.xlsx",
#'2019_CAC_sbe': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2019_sbe\CAC_2019_sbe_cropcalendars.xlsx",
#'2019_CAC_soy': r"S:\eshape\Pilot 1\data\CAC_seeds\CAC_2019_soy\CAC_2019_soy_cropcalendars.xlsx"
# '2018_Greece': r"S:\eshape\Pilot 1\data\Parcels_greece\cropCalendars2018Komotini_cropcalendars.xlsx",




# '2019_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates_overview_reduc.xlsx",
#                            '2018_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates_overview_reduc.xlsx",
#                            '2018_Flax' : r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all_overview.xlsx",

#dict_cropcalendars_data_locations = {'2019_TAP': r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.xlsx"}
# A) OPENEO data extraction
unique_ids_fields= []
df_metrics = {}
fields = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp")
for n in range(fields.shape[0]): unique_ids_fields.extend([uuid.uuid4().hex[:30].lower()])
fields['unique_ID'] = unique_ids_fields
crs = fields.crs
field_test = fields.iloc[0]
S1_Passes = [r'ASCENDING', r'DESCENDING']
start = '2019-01-01'
end = '2019-12-31'
start_year = start[0:4]
end_year = end[0:4]
start_month = start[5:7]
end_month = end[5:7]
start_day = start[8:10]
end_day = end[8:10]
connection = openeo.connect("http://openeo.vgt.vito.be/openeo/0.4.0")#openeo.connect("http://openeo.vgt.vito.be/openeo/0.4.0") #"http://openeo-dev.vgt.vito.be/openeo/1.0.0"
connection.authenticate_basic("bontek","bontek123")

#df_metrics.update(Openeo_extraction_S1_VH_VV(field_test,start,end,S1_Passes,connection))
#df_metrics.update(TSS_service_CropSAR_extraction(field_test,start,end,crs))

#### part to extract data from OpenEO
#outdir = r'S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\Test_cropsar.zip'
#df_cropsar = OpenEO_extraction_cropSAR(start, end,outdir,shp_dir)

# B) post-processing of df (plotting)

#### concatenate the cropsar df if they were not processed at once
dict_cropsar_OpenEO = dict()
dict_directory_split_file_cropsar = {'WIG_fields_2019_0_25':r"S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\2019_WIG_fields_planting_dates_0_25_perc\cropsar.csv",
                          'WIG_fields_2019_25_50': r"S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\2019_WIG_fields_planting_dates_25_50_perc\cropsar.csv",
                          'WIG_fields_2019_50_75': r"S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\2019_WIG_fields_planting_dates_50_75_perc\cropsar.csv",
                          'WIG_fields_2019_75_100':r"S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\2019_WIG_fields_planting_dates_75_100_perc\cropsar.csv"}
dict_directory_cropsar_WIG_2018 = {'WIG_fields_2018': r"S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\2018_WIG_planting_harvest_dates\cropsar.csv"}
dict_directory_cropsar_flax_2018 = {'Flax_fields_2018': r"S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR\vlas_2018_wgs_all\cropsar.csv"}
dict_cropsar_OpenEO.update({'2019_WIG':concat_df_from_Openeo(dict_directory_split_file_cropsar, data_source= 'WIG_2019')})
dict_cropsar_OpenEO.update({'2018_WIG' : concat_df_from_Openeo(dict_directory_cropsar_WIG_2018)})
dict_cropsar_OpenEO.update({'2018_Flax': concat_df_from_Openeo(dict_directory_cropsar_flax_2018)})

#### concat the cropSAR curves if they were returned from OpenEO
#dict_cropsar_OpenEO = fAPAR_CropSAR_concat_OpenEO(datasets_dict,dict_cropsar_OpenEO)
### concat the CropSAR curves if they were returned from the CropSAR webservice
CropSAR_dir = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data'
dict_cropsar_API = fAPAR_CropSAR_concat_API(datasets_dict,CropSAR_dir, country_dataset_dict)

dir_data_metrics = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data'
dict_VHVV = S1_VHVV_ratio_concat(dir_data_metrics, datasets_dict,ROs_dataset_dict)

### in case want to concatenate coherence data
# datasets_dict = {'2018_Flax': r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp",
#                  '2018_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp",
#                  '2019_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates.shp"} # for WIG_2019 all the coherence data for the fields was extracted and not only for the ones wiht planting dates like with cropsar
#dict_coherence = coherence_concat(dir_data_metrics, datasets_dict, ro_s)


# C) Plotting of the fields metrics together with its reference data on crop events

#### plot the metrics of the different fields for the different orbits of interest

#Plot_time_series_metrics_crop_calendar(datasets_dict,ro_s, metrics, dict_cropcalendars_data, crop_calendar_events, Basefolder, dict_cropsar_API, dict_VHVV, dict_coherence)

# D) Generate a DF containing the metrics per window so that it can be used to run in the NN model


#### make a 6-daily moving window for the extracted datasets and for the metrics of interest

## first make a dictionary containing the df's with the meta data on the crop events
dict_cropcalendars_data = combine_cropcalendar_data(datasets_dict, dict_cropcalendars_data_locations, crop_calendar_events)
dict_moving_window_extracts = moving_window_metrics_extraction(datasets_dict, ROs_dataset_dict, dict_VHVV, dict_cropsar_API, metrics, VH_VV_range, fAPAR_range, dict_cropcalendars_data, crop_calendar_events)

# E) Apply the NN model on the moving window extracts for different thresholds and positions exceeding the thresholds (function to define optimal model and threshold for event detection)
# RMSE plots according to threshold and position of the window exceeding the threshold used for detecting the event
Test_nr = r'Test10'
thresholds_events_detection = np.arange(0.5,1,0.02) # testing different probabilility thresholds
harvest_window_index = 2 # parameter describing the position of the 'harvest' windows which will be used to extract the predicted harvest date
window_harvest_val = 60 # parameter to define which windows will be used to validate the result. The value represents the amount of days the window may be away from the actual event
plotting_prob_detection_metrics = False
outdir_RMSE_figs = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\{}\6_daily_window_data\RMSE_plots_thresholds'.format(Test_nr)
dict_statistics_per_model = dict()
for p in range(iterations):
    Basefolder = os.path.join(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10\6_daily_window_data','iteration_{}'.format(str(p)))
    dict_model_predict_prob_window = apply_NN_model(dict_moving_window_extracts, trained_model_dir, p)
    # function to calculate the RMSE between the obsered and predicted evet per orbit and combined by using different probability threshold.
    # The final predicted event date is calculated by taking the mean dates of the windows that detected the event in a certain window around the event OR the x window that detected the event
    dict_event_predictions_date, dict_fields_no_event_detected = validate_crop_calendar_event_date(dict_cropcalendars_data, dict_model_predict_prob_window, thresholds_events_detection,window_harvest_val, harvest_window_index, crop_calendar_events, ROs_dataset_dict,p)

    # plot the RMSE against the chosen threshold for prediction
    #RMSE_plotting_against_prob_threshold(dict_event_predictions_date, ROs_dataset_dict, crop_calendar_events,outdir_RMSE_figs, country_dataset_dict, harvest_window_index,p)

    # save the amount of fields for which no event could be detected when using the x detection window for defining the event
    for dataset_no_event in dict_fields_no_event_detected:
        if p !=0:
            dict_statistics_per_model.update({dataset_no_event: pd.concat([dict_event_predictions_date.get(dataset_no_event),dict_statistics_per_model.get(dataset_no_event)])})
        else:
            dict_statistics_per_model.update({dataset_no_event: dict_event_predictions_date.get(dataset_no_event)})
        df_fields_no_event_detected = dict_fields_no_event_detected.get(dataset_no_event)
        if not os.path.exists(os.path.join(outdir_RMSE_figs,country_dataset_dict.get(dataset_no_event.rsplit('_',2)[0]), dataset_no_event.rsplit('_',2)[0], 'Fields_no_harvest_detected_window_{}_thresholds_{}_ro_combined_model_{}.csv'.format(str(harvest_window_index+1),dataset_no_event.rsplit('_',2)[0],str(p)))):
            df_fields_no_event_detected.to_csv(os.path.join(outdir_RMSE_figs,country_dataset_dict.get(dataset_no_event.rsplit('_',2)[0]), dataset_no_event.rsplit('_',2)[0], 'Fields_no_harvest_detected_window_{}_thresholds_{}_ro_combined_model_{}.csv'.format(str(harvest_window_index+1),dataset_no_event.rsplit('_',2)[0], str(p))))
    ##### function to plot the crop calendar probabilities according the NN model for the entire time series and together with the metrics
    ### use the TAP  and fields outside Belgium:
    if plotting_prob_detection_metrics:
        Plot_time_series_metrics_crop_calendar_probability(ROs_dataset_dict, metrics, dict_cropcalendars_data,crop_calendar_events, Basefolder, country_dataset_dict, dict_model_predict_prob_window, dict_cropsar_API, dict_VHVV)

##### Below some extra statistics which will be calculated like the most optimal threshold and the best model iteration
### find the best threshold and model
dict_optimal_thresholds_dataset, dict_optimal_model_dataset = find_optimal_model_threshold(dict_statistics_per_model, identifier='threshold_id')


for dataset in dict_optimal_model_dataset:
    df_optimal_model = dict_optimal_model_dataset.get(dataset)
    df_optimal_threshold= dict_optimal_thresholds_dataset.get(dataset)
    df_optimal_model.to_csv(os.path.join(outdir_RMSE_figs, country_dataset_dict.get(dataset.rsplit('_',2)[0]), 'Optimal_model_{}_window_{}.csv'.format(dataset, str(harvest_window_index+1))))
    df_optimal_threshold.to_csv(os.path.join(outdir_RMSE_figs, country_dataset_dict.get(dataset.rsplit('_',2)[0]), 'Optimal_threshold_{}_window_{}.csv'.format(dataset, str(harvest_window_index+1))))





    #### cropsar code for execute bacth: self.session\
#     .datacube_from_process("cropsar",polygon_file="/data/users/Public/vhoofk/driesj/test_100000.shp",start_date="2018-03-01",end_date="2018-08-01",**params)\
#     .execute_batch(output_file, job_options={
#     "driver-memory": "10g",
#     "driver-cores": "6",
#     "driver-memoryOverhead": "6g",
#     "executor-memory": "2g",
#     "executor-memoryOverhead": "1500m",
#     "executor-cores": "2",
#     "queue": "geoviewer"
# })




