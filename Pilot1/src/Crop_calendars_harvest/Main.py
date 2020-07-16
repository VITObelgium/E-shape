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
from shapely.geometry import  Polygon
from Pilot1.src.Crop_calendars_harvest.Main_functions import concat_df_from_Openeo
from Pilot1.src.Crop_calendars_harvest.Main_functions import fAPAR_CropSAR_concat_OpenEO
from Pilot1.src.Crop_calendars_harvest.Main_functions import fAPAR_CropSAR_concat_API
from Pilot1.src.Crop_calendars_harvest.Main_functions import S1_VHVV_ratio_concat
from Pilot1.src.Crop_calendars_harvest.Main_functions import coherence_concat
from Pilot1.src.Crop_calendars_harvest.Main_functions import Plot_time_series_metrics_crop_calendar
from Pilot1.src.Crop_calendars_harvest.Main_functions import apply_NN_model
from Pilot1.src.Crop_calendars_harvest.Main_functions import Plot_time_series_metrics_crop_calendar_probability



####### PART 1: OPENEO EXTRACTION OF THE REQUIRED METRICS + PRE-PROCESSING OF THE INPUT DATA SO THE MODEL CAN USE IT

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

### post-processing of df (plotting)

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

##### add to the df's the right field id to each column
datasets_dict = {'2018_Flax': r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp",
                 '2018_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp",
                 '2019_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_fields_planting_dates.shp"}

#### concat the cropSAR curves if they were returned from OpenEO
dict_cropsar_OpenEO = fAPAR_CropSAR_concat_OpenEO(datasets_dict,dict_cropsar_OpenEO)
### concat the CropSAR curves if they were returned from the CropSAR webservice
CropSAR_dir = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\CropSAR'
dict_cropsar_API = fAPAR_CropSAR_concat_API(datasets_dict,CropSAR_dir)

dir_data_metrics = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data'
ro_s = ['ro110','ro161']
dict_VHVV = S1_VHVV_ratio_concat(dir_data_metrics, datasets_dict,ro_s)
datasets_dict = {'2018_Flax': r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp",
                 '2018_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp",
                 '2019_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates.shp"} # for WIG_2019 all the coherence data for the fields was extracted and not only for the ones wiht planting dates like with cropsar
dict_coherence = coherence_concat(dir_data_metrics, datasets_dict, ro_s)

#### plot the metrics of the different fields for the different orbits of interest
dict_cropcalendars_data = {'2019_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates_overview_reduc.xlsx",
                           '2018_WIG': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates_overview_reduc.xlsx",
                           '2018_Flax' : r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all_overview.xlsx"}

metrics = [r'VHVV', 'CropSAR', 'coherence']
crop_calendar_events = ['Planting_date']
Basefolder = r'S:\eshape\Pilot 1\results\{}\plots'.format(crop_calendar_events[0])
#Plot_time_series_metrics_crop_calendar(datasets_dict,ro_s, metrics, dict_cropcalendars_data, crop_calendar_events, Basefolder, dict_cropsar_API, dict_VHVV, dict_coherence)

##### function that will make the dataframes ready to use in the NN model




##### function to plot the crop calendar probabilities according the NN model for the entire time series and together with the metrics
### use the TAP fields:
TAP_dict = {'2019_TAP': r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.shp"}
dict_validation_data = {'2019_TAP': pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\6_daily_window_data\df_harvest_model_6daily_TAP_only_ro110_ro161_30_daily_window.csv")}
dict_cropsar_API_TAP = fAPAR_CropSAR_concat_API(TAP_dict,CropSAR_dir)
dict_VHVV_TAP = S1_VHVV_ratio_concat(dir_data_metrics, TAP_dict,ro_s)
trained_model_dir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10'
outdir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10\6_daily_window_data'
metrics_TAP = [r'VHVV_ro110','VHVV_ro161', 'CropSAR']

#### function to use the trained model to  get the probabilities
iterations = 30
crop_calendar_events_TAP = ['harvest_da']
dict_cropcalendars_data_TAP = {'2019_TAP': r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.csv"}
for p in range(iterations):
    Basefolder = os.path.join(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10\6_daily_window_data','iteration_{}'.format(str(p)))
    dict_model_predict = apply_NN_model(dict_validation_data, trained_model_dir, outdir, p)
    Plot_time_series_metrics_crop_calendar_probability(dict_validation_data,ro_s, metrics_TAP, dict_cropcalendars_data_TAP, crop_calendar_events_TAP, Basefolder, dict_model_predict, dict_cropsar_API_TAP, dict_VHVV_TAP)





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




