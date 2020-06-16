import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
from random import choice
import random
from datetime import timedelta
dir_data = r'S:\eshape\Pilot 1\results\S1_S2_data'
Cropsar_dir = r'S:\eshape\Pilot 1\results\S1_S2_data\CropSAR'
ro_interest = 'ro110'
ids_remove = ['AWpnndUzg8l355Cx1_aC', 'AWmwd4Xj8cDLxG4hgS38', 'AWztFmWjPkIpp4CeUG4F', 'AWpe5oDKg8l355Cx0x58'] # ids to remove based on results from Test1
## Step1: Concatenate the VHVV ratio for all the fields from the different datasets
def S1_VHVV_ratio_concat(dir_data,datasets_dict, ids_remove):
    files_S1 = glob.glob(os.path.join(dir_data,'S1_*{}.csv').format(ro_interest))
    files_S1 = [item for item  in files_S1 if any(S1_pass in item for S1_pass in ['Ascending','Descending'])]
    df_S1_combined = []
    fields_ids_dataset = []
    for dataset in datasets_dict.keys():
        file_S1 = [item for item in files_S1 if dataset in item][0]
        shp = gpd.read_file(datasets_dict.get(dataset))
        field_ids = shp.id.to_list()
        field_ids = [item for item in field_ids if item not in ids_remove]
        fields_ids_dataset.extend(field_ids)

        df_S1 = pd.read_csv(file_S1)
        df_S1.Date = df_S1.Date.values.astype('datetime64[D]')
        df_S1.index = df_S1.Date
        df_S1 = df_S1.drop(columns=['Date'])
        df_S1 = df_S1.loc[:, df_S1.columns.isin([str('VV_'+item) for item in field_ids] + [str('VH_'+item) for item in field_ids])] # only the fields of interest for data extraction)]
        for id in field_ids:
            df_S1['VH_VV_{}'.format(id)] = 10*np.log(df_S1['VH_{}'.format(id)]/df_S1['VV_{}'.format(id)])
        df_S1_combined.append(df_S1)
    df_VHVV_combined = pd.concat(df_S1_combined, axis = 1)
    df_VHVV_combined = df_VHVV_combined.loc[:,df_VHVV_combined.columns.isin(['VH_VV_' + item for item in fields_ids_dataset])]

    return df_VHVV_combined,fields_ids_dataset
datasets_dict = {'Flax':r"S:\eshape\Pilot 1\results\Training_Val_selection\2018_Flax_fields_selected.shp",'2018_WIG':r"S:\eshape\Pilot 1\results\Training_Val_selection\2018_WIG_fields_selected.shp",
                     '2019_WIG':r"S:\eshape\Pilot 1\results\Training_Val_selection\2019_WIG_fields_selected.shp",'TAP':r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.shp"}
df_VHVV_combined, fields_dataset_compiling = S1_VHVV_ratio_concat(dir_data,datasets_dict,ids_remove)

## STEP2: Concatenate the fAPAR for all the fields from the different datasets
def fAPAR_CropSAR_concat(Cropsar_dir,df_VHVV_combined):
    files_cropsar = glob.glob(os.path.join(Cropsar_dir,'*_cropsar.csv'))
    df_cropsar_combined = []
    for file_cropsar in files_cropsar:
        df_cropsar = pd.read_csv(file_cropsar)
        df_cropsar = df_cropsar.rename(columns={'Unnamed: 0': 'Date'})
        df_cropsar.Date = df_cropsar.Date.values.astype('datetime64[D]')
        df_cropsar.index = df_cropsar.Date
        df_cropsar = df_cropsar.drop(columns=['Date'])
        df_cropsar = pd.DataFrame(df_cropsar['q50'])
        if 'TAP' in file_cropsar:
            df_cropsar = df_cropsar.rename(columns = {'q50':'{}'.format(os.path.split(file_cropsar)[-1].split('parcel_TAP_Monitoring_fields_')[-1].split('_cropsar')[0])})
        else:
            df_cropsar = df_cropsar.rename(columns = {'q50':'{}'.format(os.path.split(file_cropsar)[-1].split('parcel_')[-1].split('_cropsar')[0])})
        df_cropsar_combined.append(df_cropsar)
    df_cropsar_combined = pd.concat(df_cropsar_combined,axis = 1)
    df_cropsar_combined = df_cropsar_combined.reindex(df_VHVV_combined.index, method = 'nearest')
    return df_cropsar_combined
df_cropsar_combined = fAPAR_CropSAR_concat(Cropsar_dir, df_VHVV_combined)
df_cropsar_combined = df_cropsar_combined.loc[:,df_cropsar_combined.columns.isin(fields_dataset_compiling)] ## subset the dataset based on the fields of interest

#### STEP3 Define the harvest date of the selected fields
 #a) Load the df's containing the harvest date information
def harvest_dates_concat(datasets_dict):

    df_harvest_dates = []
    for dataset in datasets_dict:
        df_meta_harvest_dates = pd.DataFrame(gpd.read_file(datasets_dict.get(dataset)))
        if dataset == 'Flax':
            df_meta_harvest_dates = df_meta_harvest_dates.rename(columns={'Tijdstip v': 'Harvest_date'})
            df_meta_harvest_dates['Harvest_date'] =  pd.to_datetime(df_meta_harvest_dates['Harvest_date'].values,format = '%d/%m/%Y')
        else:
            df_meta_harvest_dates = df_meta_harvest_dates.rename(columns={'harvest_da': 'Harvest_date'})
            df_meta_harvest_dates['Harvest_date'] =  df_meta_harvest_dates['Harvest_date'].values.astype('datetime64[D]')

        df_harvest_dates.append(df_meta_harvest_dates)
    df_harvest_dates = pd.concat(df_harvest_dates, axis = 0)
    return df_harvest_dates
df_harvest_dates = harvest_dates_concat(datasets_dict)

# b) create df which serves as input data for the model
def compile_harvest_model_data(df_cropsar_combined,df_VHVV_combined,fields_dataset_compiling, df_harvest_dates):
    moving_window_steps = np.arange(0,df_cropsar_combined.shape[0]-3)
    window_values = 4 # 4 coverage are extracted within the window
    temporal_res = 6 # the resolution of the data in days
    window_width = (window_values-1)*6 #days within the window
    df_harvest_model = []
    print('{} FIELDS TO COMPILE IN DATASET'.format(len(fields_dataset_compiling)-1))
    for id in fields_dataset_compiling:
        print('FIELD {}: COMPILING OF {}'.format(fields_dataset_compiling.index(id),id))
        df_cropsar_field = df_cropsar_combined.loc[:, df_cropsar_combined.columns.isin([id])]
        df_VHVV_field = df_VHVV_combined.loc[:, df_VHVV_combined.columns.isin(['VH_VV_'+id])]
        df_field_input_data_harvest_model = []
        for p in range(len(moving_window_steps)):
            ### sample some fAPAR data within the window and derive the difference from the harves date for the selected window
            df_cropsar_field_window = pd.DataFrame(df_cropsar_field.iloc[p:p+4,0])
            if df_cropsar_field_window.isnull().values.any() or df_cropsar_field_window.isnull().values.any():
                continue
            prediction_date_window =  pd.DataFrame(df_cropsar_field_window.index[0] + timedelta(window_width/2), index = [id+'_{}'.format(str(p))], columns=(['prediction_date_window']))#the center data of the window which can is in fact the harvest prediction date if the model returns 1
            harvest_date_field = pd.DataFrame([df_harvest_dates.loc[df_harvest_dates.id == id]['Harvest_date'].values[0]] , index = [id+'_{}'.format(str(p))], columns=['Harvest_date'])
            df_cropsar_field_window = pd.DataFrame(df_cropsar_field_window.T.values, index = [id+'_{}'.format(str(p))], columns= (['fAPAR_{}'.format(n) for n in range(1,5)]))
            #df_cropsar_field = df_cropsar_field.reset_index()

            df_VHVV_field_window = pd.DataFrame(df_VHVV_field.iloc[p:p+4,0])
            if df_VHVV_field_window.isnull().values.any() or df_VHVV_field_window.isna().values.any():
                continue
            df_VHVV_field_window.index.name = 'Date'
            df_dates_window = pd.DataFrame(df_VHVV_field_window.index)
            df_dates_window = pd.DataFrame(df_dates_window.T.values, index = [id+'_{}'.format(str(p))], columns= (['Date_{}'.format(n) for n in range(1,5)]))
            if pd.isnull(df_harvest_dates.loc[df_harvest_dates.id == id]['Harvest_date'].values[0]):
                continue
            df_diff_harvest_window = pd.DataFrame(df_VHVV_field_window.index-df_harvest_dates.loc[df_harvest_dates.id == id]['Harvest_date'].values[0])
            df_diff_harvest_window = pd.DataFrame(df_diff_harvest_window.T.values, index = [id+'_{}'.format(str(p))], columns= (['Diff_harvest_{}'.format(n) for n in range(1,5)]))
            df_VHVV_field_window = pd.DataFrame(df_VHVV_field_window.T.values, index = [id+'_{}'.format(str(p))], columns= (['ro110_VHVV_{}'.format(n) for n in range(1,5)]))
            df_window_field_concat_var = pd.concat([df_cropsar_field_window,df_VHVV_field_window,df_diff_harvest_window,df_dates_window,prediction_date_window, harvest_date_field], axis = 1)
            df_field_input_data_harvest_model.append(df_window_field_concat_var)
        if df_field_input_data_harvest_model:
            df_field_input_data_harvest_model = pd.concat(df_field_input_data_harvest_model, axis = 0)
            df_harvest_model.append(df_field_input_data_harvest_model)
    df_harvest_model = pd.concat(df_harvest_model, axis = 0)
    df_harvest_model.index.name = 'ID_field'
    return df_harvest_model
df_harvest_model = compile_harvest_model_data(df_cropsar_combined,df_VHVV_combined,fields_dataset_compiling, df_harvest_dates)
df_harvest_model.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\6_daily_window_data\df_harvest_model_6daily.csv')


