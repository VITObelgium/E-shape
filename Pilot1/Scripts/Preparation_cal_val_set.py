import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
from random import choice
import random
dir_data = r'S:\eshape\Pilot 1\results\S1_S2_data'
Cropsar_data = r'S:\eshape\Pilot 1\results\S1_S2_data\CropSAR'
ro_interest = 'ro110'
Test_nr = r'Test4'
metrics_exclude = ['slope','coh']
##### Step 0: Prepare the raw dataframes so that they can be used to extract data around the harvest
# a)Filter the coherence df so only the VV coherence is used
files_coherence = glob.glob(os.path.join(dir_data,'S1_coherence_*_dates_{}.csv').format(ro_interest))
df_coherence_filter = []
for file_coherence in files_coherence:
    if 'Flax' in file_coherence:
        shp = gpd.read_file(r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp")
    elif 'WIG' and '2018' in file_coherence:
        shp = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp")
    elif  'WIG' and '2019' in file_coherence:
        shp = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates.shp")
    field_ids = shp.id.to_list()
    vh_ids = np.arange(0.1, len(field_ids),1)
    vh_ids = [str(item) for item in vh_ids]
    df  = pd.read_csv(file_coherence,skiprows= range(1,3))
    df = df.rename(columns={'polygon': 'Date'})
    df.Date = df.Date.values.astype('datetime64[D]')
    df.index = df.Date
    df = df.drop(columns=['Date']+list(vh_ids)) ### only use the VV coherence (more sensitive)
    #df  = df.interpolate(method = 'time') # avoid data gaps, linear interpolate based on time
    df.columns = field_ids
    df = df*0.004 # rescaling the coherence to values between 0 and 1
    df_coherence_filter.append(df)
df_coherence_filter = pd.concat(df_coherence_filter, axis = 1)

# b)Prepare the original S1 dataframe so that the VH/VV ratio can be calculated
files_S1 = glob.glob(os.path.join(dir_data,'S1_*_dates_{}.csv').format(ro_interest))
files_S1 = [item for item  in files_S1 if any(S1_pass in item for S1_pass in ['Ascending','Descending'])]
df_S1_combined = []
for file_S1 in files_S1:
    if 'Flax' in file_S1:
        shp = gpd.read_file(r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp")
    elif 'WIG' and '2018' in file_S1:
        shp = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates.shp")
    elif 'WIG' and '2019' in file_S1:
        shp = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates.shp")
    field_ids = shp.id.to_list()
    field_ids = [str(item) for item in field_ids]

    df_S1 = pd.read_csv(file_S1)
    df_S1.Date = df_S1.Date.values.astype('datetime64[D]')
    df_S1.index = df_S1.Date
    df_S1 = df_S1.drop(columns=['Date'])
    #df_S1 = df_S1.interpolate(method = 'time') # avoid data gaps, take last valid observation to fill
    for id in field_ids:
        df_S1['VH_VV_{}'.format(id)] = 10*np.log(df_S1['VH_{}'.format(id)]/df_S1['VV_{}'.format(id)])
    df_S1_combined.append(df_S1)
df_S1_combined = pd.concat(df_S1_combined, axis = 1)
# c)Create one df with the fAPAR for all fields
files_cropsar = glob.glob(os.path.join(Cropsar_data,'*_cropsar.csv'))
df_cropsar_combined = []
df_slope_fAPAR_combined = []
for file_cropsar in files_cropsar:
    df_cropsar = pd.read_csv(file_cropsar)
    df_cropsar = df_cropsar.rename(columns={'Unnamed: 0': 'Date'})
    df_cropsar.Date = df_cropsar.Date.values.astype('datetime64[D]')
    df_cropsar.index = df_cropsar.Date
    df_cropsar = df_cropsar.drop(columns=['Date'])
    df_cropsar = pd.DataFrame(df_cropsar['q50'])
    df_slope_fAPAR = pd.DataFrame(np.gradient(df_cropsar['q50']),index = df_cropsar.index, columns=(['slope']))
    df_cropsar = df_cropsar.rename(columns = {'q50':'{}'.format(os.path.split(file_cropsar)[-1].split('parcel_')[-1].split('_cropsar')[0])})
    df_slope_fAPAR = df_slope_fAPAR.rename(columns = {'slope':'{}'.format(os.path.split(file_cropsar)[-1].split('parcel_')[-1].split('_cropsar')[0])})
    df_cropsar_combined.append(df_cropsar)
    df_slope_fAPAR_combined.append(df_slope_fAPAR)
df_cropsar_combined = pd.concat(df_cropsar_combined,axis = 1)
df_slope_fAPAR_combined = pd.concat(df_slope_fAPAR_combined, axis= 1)
##### Step 1: Filter the df's containing coherence, fAPAR, VH/VV to the selected fields for cal/val
# a) extract all the id's which are going to be used for cal/val
shp_flax = gpd.read_file(r"S:\eshape\Pilot 1\results\Training_Val_selection\2018_Flax_fields_selected.shp")
ids_flax_selected = shp_flax.id.to_list()
shp_WIG_2018 = gpd.read_file(r"S:\eshape\Pilot 1\results\Training_Val_selection\2018_WIG_fields_selected.shp")
ids_WIG_2018 = shp_WIG_2018.id.to_list()
shp_WIG_2019 = gpd.read_file(r"S:\eshape\Pilot 1\results\Training_Val_selection\2019_WIG_fields_selected.shp")
ids_WIG_2019 = shp_WIG_2019.id.to_list()

ids_cal_val = ids_flax_selected + ids_WIG_2018 + ids_WIG_2019
ids_remove = ['AWpnndUzg8l355Cx1_aC', 'AWmwd4Xj8cDLxG4hgS38', 'AWztFmWjPkIpp4CeUG4F', 'AWpe5oDKg8l355Cx0x58'] # ids to remove based on results from Test1
ids_cal_val = [item for item in ids_cal_val if item not in ids_remove]

df_coherence_fields_selected = df_coherence_filter.loc[:,df_coherence_filter.columns.isin(ids_cal_val)]
df_VHVV_ratio_fields_selected = df_S1_combined.loc[:,df_S1_combined.columns.isin(['VH_VV_'+item for item in ids_cal_val])]
df_cropsar_fields_selected = df_cropsar_combined.loc[:,df_cropsar_combined.columns.isin(ids_cal_val)]
df_cropsar_fields_selected = df_cropsar_fields_selected.reindex(df_coherence_fields_selected.index, method= 'nearest') # set that the fAPAR df to the same temporal resolution as for coherence and S1
df_slope_fAPAR_fields_selected = df_slope_fAPAR_combined.loc[:,df_slope_fAPAR_combined.columns.isin(ids_cal_val)]
df_slope_fAPAR_fields_selected = df_slope_fAPAR_fields_selected.reindex(df_coherence_fields_selected.index, method= 'nearest')

#### b) Extract the data around the harvest date + select some periods away from the harvest
df_meta_flax = pd.read_excel(r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all_overview.xlsx")
df_meta_WIG_2018 = pd.read_excel(r"S:\eshape\Pilot 1\data\WIG_data\{}_WIG_planting_harvest_dates_overview.xlsx".format(str(2018)), sheet_name='{}_WIG_planting_harvest_dates'.format(str(2018)))
df_meta_WIG_2019 = pd.read_excel(r"S:\eshape\Pilot 1\data\WIG_data\{}_WIG_planting_harvest_dates_overview.xlsx".format(str(2019)), sheet_name='{}_WIG_planting_harvest_dates'.format(str(2019)))

datasets = ['calibration','validation']

iterations = 100
for p in range(iterations):

    random.shuffle(ids_cal_val) ### so that a random subset of fields will be chosen for validation and calibration
    for dataset in datasets:
        if 'calibration' in dataset:
            ids_dataset = ids_cal_val[0: int(len(ids_cal_val)*0.7)]
        else:
            ids_dataset = ids_cal_val[int(len(ids_cal_val)*0.7):]

        df_harvest_data_extraction = []
        df_non_harvest_data_extraction = []
        for id_cal_val in ids_dataset:
            if id_cal_val in ids_flax_selected:
                df_harvest_date = df_meta_flax.loc[df_meta_flax.id == id_cal_val]['Tijdstip v'] #take into account that it is actually mowing date for the flax fields
                df_harvest_date =pd.to_datetime(df_harvest_date.values,format = '%d/%m/%Y')

            elif id_cal_val in ids_WIG_2019:
                df_harvest_date = df_meta_WIG_2019.loc[df_meta_WIG_2019.id == id_cal_val]['harvest_da']
                df_harvest_date = pd.to_datetime(df_harvest_date.values)

            elif id_cal_val in ids_WIG_2018:
                df_harvest_date = df_meta_WIG_2018.loc[df_meta_WIG_2018.id == id_cal_val]['harvest_da']
                df_harvest_date = pd.to_datetime(df_harvest_date.values)
        ### calculate the difference in days with the harvest date

            # coherence
            df_diff_harvest_coherence= pd.DataFrame(df_coherence_fields_selected.loc[:,df_coherence_fields_selected.columns.isin([id_cal_val])].index- df_harvest_date.values[0])
            df_diff_harvest_coherence.index = df_coherence_fields_selected.index
            df_diff_harvest_coherence.sort_index(ascending=True, inplace=True)
            df_harvest_values_field_coherence = df_coherence_fields_selected.loc[:,df_coherence_fields_selected.columns.isin([id_cal_val])][df_diff_harvest_coherence.Date <= pd.to_timedelta(0)].iloc[-2:] ### Extraction of coherence for two closest dates before harvest

            df_harvest_values_field_coherence = df_harvest_values_field_coherence.append(df_coherence_fields_selected.loc[:,df_coherence_fields_selected.columns.isin([id_cal_val])][df_diff_harvest_coherence.Date > pd.to_timedelta(0)].iloc[0:2])  ### Extraction of coherence for two closest dates after harvest
            df_harvest_values_field_coherence = df_harvest_values_field_coherence.rename(columns = {'{}'.format(id_cal_val):'{}_coherence_ro110'.format(id_cal_val)})
            if df_harvest_values_field_coherence.isnull().values.any() or df_harvest_values_field_coherence.isna().values.any():  # avoid having data gaps
                continue
            #fAPAR
            df_diff_harvest_cropsar = pd.DataFrame(df_cropsar_fields_selected.loc[:,df_cropsar_fields_selected.columns.isin([id_cal_val])].index-df_harvest_date.values[0])
            df_diff_harvest_cropsar.index = df_cropsar_fields_selected.index
            df_diff_harvest_cropsar.sort_index(ascending= True, inplace=True)
            df_harvest_values_field_cropsar = df_cropsar_fields_selected.loc[:,df_cropsar_fields_selected.columns.isin([id_cal_val])][df_diff_harvest_cropsar.Date <= pd.to_timedelta(0)].iloc[-2:] ### Extraction of fAPAR for two closest dates before harvest
            df_harvest_values_field_cropsar = df_harvest_values_field_cropsar.append(df_cropsar_fields_selected.loc[:,df_cropsar_fields_selected.columns.isin([id_cal_val])][df_diff_harvest_cropsar.Date > pd.to_timedelta(0)].iloc[0:2]) ### Extraction of fAPAR for two closest dates after harvest
            df_harvest_values_field_cropsar = df_harvest_values_field_cropsar.rename(columns = {'{}'.format(id_cal_val):'{}_fAPAR'.format(id_cal_val)})

            # slope fAPAR
            df_diff_harvest_slope_fAPAR = pd.DataFrame(df_slope_fAPAR_fields_selected.loc[:,df_slope_fAPAR_fields_selected.columns.isin(ids_cal_val)].index-df_harvest_date.values[0])
            df_diff_harvest_slope_fAPAR.index = df_slope_fAPAR_fields_selected.index
            df_diff_harvest_slope_fAPAR.sort_index(ascending= True, inplace= True)
            df_harvest_values_field_slope_fAPAR = df_slope_fAPAR_fields_selected.loc[:,df_slope_fAPAR_fields_selected.columns.isin([id_cal_val])][df_diff_harvest_slope_fAPAR.Date <= pd.to_timedelta(0)].iloc[-2:]  ### Extraction of fAPAR for two closest dates before harvest
            df_harvest_values_field_slope_fAPAR = df_harvest_values_field_slope_fAPAR.append(df_slope_fAPAR_fields_selected.loc[:,df_slope_fAPAR_fields_selected.columns.isin([id_cal_val])][df_diff_harvest_slope_fAPAR.Date > pd.to_timedelta(0)].iloc[0:2])  ### Extraction of fAPAR for two closest dates after harvest
            df_harvest_values_field_slope_fAPAR = df_harvest_values_field_slope_fAPAR.rename(columns = {'{}'.format(id_cal_val):'{}_slope_fAPAR'.format(id_cal_val)})
            ## VH/VV ratio
            df_diff_harvest_vhvv = pd.DataFrame(df_VHVV_ratio_fields_selected.loc[:,df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_'+id_cal_val])].index-df_harvest_date.values[0])
            df_diff_harvest_vhvv.index = df_VHVV_ratio_fields_selected.index
            df_diff_harvest_vhvv.sort_index(ascending=True, inplace=True)
            df_harvest_values_field_vhvv = df_VHVV_ratio_fields_selected.loc[:,df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_'+id_cal_val])][df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].iloc[-2:] ### Extraction of VHVV for two closest dates before harvest

            df_harvest_values_field_vhvv = df_harvest_values_field_vhvv.append(df_VHVV_ratio_fields_selected.loc[:,df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_'+id_cal_val])][df_diff_harvest_vhvv.Date > pd.to_timedelta(0)].iloc[0:2]) ### Extraction of VHVV for two closest dates after harvest

            df_harvest_values_field_vhvv = df_harvest_values_field_vhvv.rename(columns = {'{}'.format(id_cal_val):'{}_VHVV'.format(id_cal_val)})

            if df_harvest_values_field_vhvv.isnull().values.any() or df_harvest_values_field_vhvv.isna().values.any():  # avoid having data gaps
                continue
            ### combine all the S1S2 metrics in one df per field
            df_harvest_data_extraction.append(pd.concat([pd.DataFrame(df_harvest_values_field_coherence.T.values, index = [id_cal_val], columns = (['ro110_coh_{}'.format(n) for n in range(1,5)])),
                             pd.DataFrame(df_harvest_values_field_cropsar.T.values, index = [id_cal_val], columns = (['fAPAR_{}'.format(n) for n in range(1,5)])),
                             pd.DataFrame(df_harvest_values_field_vhvv.T.values, index = [id_cal_val], columns = (['ro110_VHVV_{}'.format(n) for n in range(1,5)])),  pd.DataFrame(df_harvest_values_field_slope_fAPAR.T.values, index = [id_cal_val], columns = (['slope_fAPAR_{}'.format(n) for n in range(1,5)]))], axis=1))


            #### part to extract NON-harvest data
            ## 3 periods after harvest extraction
            counter_after_harvest_select = 0
            random_sample_excl = []
            date_range_after_harvest = pd.date_range(start = (df_VHVV_ratio_fields_selected.loc[:, df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_diff_harvest_vhvv.Date > pd.to_timedelta(0)].dropna().iloc[3:].index[0].strftime('%Y-%m-%d')), end = (df_VHVV_ratio_fields_selected.loc[:, df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_diff_harvest_vhvv.Date > pd.to_timedelta(0)].dropna().iloc[-1:].index[0].strftime('%Y-%m-%d')), freq='6D')
            random_sample = [0]
            ## pick random date out of date range
            while random_sample and counter_after_harvest_select <3:
                try:
                    random_sample = choice([i for i in range(1, len(date_range_after_harvest) - 3) if i not in random_sample_excl]) # see that the boundary of the range are excluded for period selection
                except:
                    random_sample = [] # no sampling possibilities left
                if not random_sample:
                    continue
                df_no_harvest_values_field_vhvv = df_VHVV_ratio_fields_selected.loc[:,df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_'+id_cal_val])][df_VHVV_ratio_fields_selected.index >= date_range_after_harvest[random_sample-1]].iloc[0:4] # take all the harvest values around the selected date
                if df_no_harvest_values_field_vhvv.isnull().values.any() or df_no_harvest_values_field_vhvv.isna().values.any():
                    random_sample_excl.extend([random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3])
                    continue
                df_no_harvest_values_field_coherence = df_coherence_fields_selected.loc[:,df_coherence_fields_selected.columns.isin([id_cal_val])][df_coherence_fields_selected.index >= date_range_after_harvest[random_sample-1]].iloc[0:4]
                if df_no_harvest_values_field_coherence.isnull().values.any() or df_no_harvest_values_field_coherence.isna().values.any():
                    random_sample_excl.extend([random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3])
                    continue
                df_no_harvest_values_field_cropsar = df_cropsar_fields_selected.loc[:,df_cropsar_fields_selected.columns.isin([id_cal_val])][df_cropsar_fields_selected.index >= date_range_after_harvest[random_sample-1]].iloc[0:4]

                df_no_harvest_values_field_slope_fAPAR = df_slope_fAPAR_fields_selected.loc[:,df_slope_fAPAR_fields_selected.columns.isin([id_cal_val])][df_slope_fAPAR_fields_selected.index >= date_range_after_harvest[random_sample-1]].iloc[0:4]
                df_non_harvest_data_extraction.append(pd.concat([pd.DataFrame(df_no_harvest_values_field_coherence.T.values,index=[id_cal_val+'_after_{}'.format(str(counter_after_harvest_select))], columns=(['ro110_coh_{}'.format(n) for n in range(1, 5)])),
                                                        pd.DataFrame(df_no_harvest_values_field_cropsar.T.values, index=[id_cal_val+'_after_{}'.format(str(counter_after_harvest_select))], columns=(['fAPAR_{}'.format(n) for n in range(1, 5)])),
                                                             pd.DataFrame(df_no_harvest_values_field_vhvv.T.values,index=[id_cal_val+'_after_{}'.format(str(counter_after_harvest_select))], columns=(['ro110_VHVV_{}'.format(n) for n in range(1, 5)])),
                                                                 pd.DataFrame(df_no_harvest_values_field_slope_fAPAR.T.values, index=[id_cal_val+'_after_{}'.format(str(counter_after_harvest_select))], columns=(['slope_fAPAR_{}'.format(n) for n in range(1, 5)]))], axis=1))
                random_sample_excl.extend([random_sample-3, random_sample -2, random_sample-1, random_sample, random_sample+1, random_sample+2, random_sample +3])
                counter_after_harvest_select += 1

            ## 3 periods before harvest extraction
            counter_before_harvest_select = 0
            random_sample_excl = []
            date_range_before_harvest = pd.date_range(start=(df_VHVV_ratio_fields_selected.loc[:, df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].dropna().iloc[:-2].index[0].strftime('%Y-%m-%d')),
                                                      end=(df_VHVV_ratio_fields_selected.loc[:, df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].dropna().iloc[:-2].index[-1].strftime('%Y-%m-%d')),
                                                     freq='6D')
            if df_VHVV_ratio_fields_selected.loc[:, df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].dropna().iloc[:-2].index[0].month <5:
                date_range_before_harvest = pd.date_range(start='{}-05-01'.format(str(df_harvest_date.year.values[0])),
                                                          end=(df_VHVV_ratio_fields_selected.loc[:,df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_diff_harvest_vhvv.Date <= pd.to_timedelta(
                                                                       0)].dropna().iloc[:-2].index[-1].strftime('%Y-%m-%d')),
                                                          freq='6D')
            random_sample = [0]
            while random_sample and counter_before_harvest_select < 3:
                try:
                    random_sample = choice([i for i in range(1, len(date_range_before_harvest) - 3) if i not in random_sample_excl])  # see that the boundary of the range are excluded for period selection
                except:
                    random_sample = [] # no sampling possibilities left
                if not random_sample:
                    continue
                df_no_harvest_values_field_vhvv = df_VHVV_ratio_fields_selected.loc[:, df_VHVV_ratio_fields_selected.columns.isin(['VH_VV_' + id_cal_val])][df_VHVV_ratio_fields_selected.index >= date_range_before_harvest[random_sample - 1]].iloc[0:4]  # take all the harvest values around the selected date
                if df_no_harvest_values_field_vhvv.isnull().values.any() or df_no_harvest_values_field_vhvv.isna().values.any():
                    random_sample_excl.extend([random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3])
                    continue
                df_no_harvest_values_field_coherence = df_coherence_fields_selected.loc[:, df_coherence_fields_selected.columns.isin([id_cal_val])][df_coherence_fields_selected.index >= date_range_before_harvest[random_sample - 1]].iloc[0:4]
                if df_no_harvest_values_field_coherence.isnull().values.any() or df_no_harvest_values_field_coherence.isna().values.any():
                    random_sample_excl.extend([random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3])
                    continue
                df_no_harvest_values_field_cropsar = df_cropsar_fields_selected.loc[:, df_cropsar_fields_selected.columns.isin([id_cal_val])][df_cropsar_fields_selected.index >= date_range_before_harvest[random_sample - 1]].iloc[0:4]
                df_no_harvest_values_field_slope_fAPAR = df_slope_fAPAR_fields_selected.loc[:,df_slope_fAPAR_fields_selected.columns.isin([id_cal_val])][df_slope_fAPAR_fields_selected.index >= date_range_before_harvest[random_sample-1]].iloc[0:4]
                df_non_harvest_data_extraction.append(pd.concat([pd.DataFrame(df_no_harvest_values_field_coherence.T.values,index=[id_cal_val + '_before_{}'.format(str(counter_before_harvest_select))], columns=(['ro110_coh_{}'.format(n) for n in range(1, 5)])),
                                                                 pd.DataFrame(df_no_harvest_values_field_cropsar.T.values,index=[id_cal_val + '_before_{}'.format(str(counter_before_harvest_select))], columns=(['fAPAR_{}'.format(n) for n in range(1, 5)])),
                                                                 pd.DataFrame(df_no_harvest_values_field_vhvv.T.values, index=[id_cal_val + '_before_{}'.format(str(counter_before_harvest_select))], columns=(['ro110_VHVV_{}'.format(n) for n in range(1, 5)])),
                                                                 pd.DataFrame(df_no_harvest_values_field_slope_fAPAR.T.values, index=[id_cal_val+'_before_{}'.format(str(counter_before_harvest_select))], columns=(['slope_fAPAR_{}'.format(n) for n in range(1, 5)]))], axis=1))
                random_sample_excl.extend([random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3])
                counter_before_harvest_select += 1



        #### concat all the df with no and_harvest data
        df_harvest_data_extraction = pd.concat(df_harvest_data_extraction)
        df_non_harvest_data_extraction = pd.concat(df_non_harvest_data_extraction)

        ## remove metrics if specified
        if metrics_exclude:
            for s in range(len(metrics_exclude)):
                df_harvest_data_extraction.drop([col for col in df_harvest_data_extraction if metrics_exclude[s] in col], axis=1, inplace=True)
                df_non_harvest_data_extraction.drop([col for col in df_non_harvest_data_extraction if metrics_exclude[s] in col], axis = 1, inplace = True)

        ## add the y-column which will tell the neural network if it is harvest data or not (1 or 0)
        y_harvest = [1]*df_harvest_data_extraction.index.size ### columns indicatin that the extracted data is at the period of harvest
        y_no_harvest = [0]*df_non_harvest_data_extraction.index.size ### columns indicating that the extracted data is at the period of no harvest
        df_harvest_data_extraction['y'] = y_harvest
        df_non_harvest_data_extraction['y'] = y_no_harvest

        if 'calibration' in dataset:
            df_calibration = pd.concat([df_harvest_data_extraction, df_non_harvest_data_extraction])
            df_calibration.index.name = 'ID_field'
            if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\calibration\{}'.format(Test_nr)): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\calibration\{}'.format(Test_nr))
            df_calibration.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\calibration\{}\df_calibration_coh_VHVV_fAPAR_update1.0_iteration{}.csv'.format(Test_nr,str(p)))
        else:
            df_validation = pd.concat([df_harvest_data_extraction, df_non_harvest_data_extraction])
            df_validation.index.name = 'ID_field'
            if not os.path.exists(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\{}'.format(Test_nr)): os.makedirs(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\{}'.format(Test_nr))
            df_validation.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\validation\{}\df_validation_coh_VHVV_fAPAR_update1.0_iteration{}.csv'.format(Test_nr,str(p)))