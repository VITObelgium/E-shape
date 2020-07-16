import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
from random import choice
import random

dir_data = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data'
Cropsar_dir = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\CropSAR'
ro_interest = 'ro110'
ids_remove = ['AWpnndUzg8l355Cx1_aC', 'AWmwd4Xj8cDLxG4hgS38', 'AWztFmWjPkIpp4CeUG4F', 'AWpe5oDKg8l355Cx0x58'] # ids to remove based on results from Test1
metrics_exclude = ['coh', 'slope', 'fAPAR']
ro_s = ['ro110','ro161']
Test_nr = r'Test12'
outdir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\test\{}'.format(Test_nr)
if not os.path.exists(outdir):os.makedirs(outdir)
df_test_harvest = []
df_test_non_harvest = []
Year_interest = 2019
VH_VV_range = [-30,-8]
fAPAR_range = [0,1]
values_before_harvest = 3
values_after_harvest = 2
for ro_interest in ro_s:
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
        # rescale to -1 and 1
        df_VHVV_combined = 2 * (df_VHVV_combined - VH_VV_range[0]) / (VH_VV_range[1] - VH_VV_range[0]) - 1

        return df_VHVV_combined,fields_ids_dataset
    datasets_dict = {'TAP':r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.shp"}
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
        # rescale to -1 and 1
        df_cropsar_combined = 2*(df_cropsar_combined-fAPAR_range[0])/ (fAPAR_range[1]- fAPAR_range[0])-1
        df_cropsar_combined = df_cropsar_combined.reindex(df_VHVV_combined.index, method = 'nearest')
        return df_cropsar_combined
    df_cropsar_combined = fAPAR_CropSAR_concat(Cropsar_dir, df_VHVV_combined)
    df_cropsar_combined = df_cropsar_combined.loc[:,df_cropsar_combined.columns.isin(fields_dataset_compiling)] ## subset the dataset based on the fields of interest


    #### STEP3 Define the harvest date of the selected fields
    # a) Load the df's containing the harvest date information
    def harvest_dates_concat(datasets_dict):

        df_harvest_dates = []
        for dataset in datasets_dict:
            df_meta_harvest_dates = pd.DataFrame(gpd.read_file(datasets_dict.get(dataset)))
            if dataset == 'Flax':
                df_meta_harvest_dates = df_meta_harvest_dates.rename(columns={'Tijdstip v': 'Harvest_date'})
                df_meta_harvest_dates['Harvest_date'] = pd.to_datetime(df_meta_harvest_dates['Harvest_date'].values,
                                                                       format='%d/%m/%Y')
            else:
                df_meta_harvest_dates = df_meta_harvest_dates.rename(columns={'harvest_da': 'Harvest_date'})
                df_meta_harvest_dates['Harvest_date'] = df_meta_harvest_dates['Harvest_date'].values.astype(
                    'datetime64[D]')

            df_harvest_dates.append(df_meta_harvest_dates)
        df_harvest_dates = pd.concat(df_harvest_dates, axis=0)
        return df_harvest_dates
    df_harvest_dates = harvest_dates_concat(datasets_dict)


    # b) create df which serves as input data for the model
    def compile_harvest_model_data(df_cropsar_combined, df_VHVV_combined, fields_dataset_compiling, df_harvest_dates, Year_interest):
        df_harvest_model_window = []
        print('{} FIELDS TO COMPILE IN DATASET'.format(len(fields_dataset_compiling) - 1))
        for id in fields_dataset_compiling:
            df_harvest_date = df_harvest_dates.loc[df_harvest_dates.id == id]['Harvest_date']
            df_harvest_date = pd.to_datetime(df_harvest_date.values)
            if df_harvest_date.isnull() or df_harvest_date.year[0] != Year_interest:
                continue
            print('FIELD {}: COMPILING OF {}'.format(fields_dataset_compiling.index(id), id))
            df_cropsar_field = df_cropsar_combined.loc[:, df_cropsar_combined.columns.isin([id])]
            df_VHVV_field = df_VHVV_combined.loc[:, df_VHVV_combined.columns.isin(['VH_VV_' + id])]

            ### sample some fAPAR data within the window and derive the difference from the harvest date for the selected window
            df_diff_harvest_cropsar = pd.DataFrame(df_cropsar_field.index -df_harvest_date.values[0])
            df_diff_harvest_cropsar.index = df_cropsar_field.index
            df_diff_harvest_cropsar.sort_index(ascending=True, inplace=True)
            df_harvest_values_field_cropsar = df_cropsar_field[df_diff_harvest_cropsar.Date <= pd.to_timedelta(0)].iloc[-values_before_harvest:]  ### Extraction of fAPAR for two closest dates before harvest
            df_harvest_values_field_cropsar = df_harvest_values_field_cropsar.append(df_cropsar_field[df_diff_harvest_cropsar.Date > pd.to_timedelta(0)].iloc[0:values_after_harvest])  ### Extraction of fAPAR for two closest dates after harvest
            df_harvest_values_field_cropsar = df_harvest_values_field_cropsar.rename(columns={'{}'.format(id): '{}_fAPAR'.format(id)})
            if df_harvest_values_field_cropsar.isnull().values.any() or df_harvest_values_field_cropsar.isnull().values.any():
                continue
            ## VH/VV ratio
            df_diff_harvest_vhvv = pd.DataFrame(df_VHVV_field.loc[:,df_VHVV_field.columns.isin(['VH_VV_' + id])].index - df_harvest_date.values[0])
            df_diff_harvest_vhvv.index = df_VHVV_field.index
            df_diff_harvest_vhvv.sort_index(ascending=True, inplace=True)
            df_harvest_values_field_vhvv = df_VHVV_field.loc[:, df_VHVV_field.columns.isin(['VH_VV_' + id])][df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].iloc[-values_before_harvest:]  ### Extraction of VHVV for two closest dates before harvest

            df_harvest_values_field_vhvv = df_harvest_values_field_vhvv.append(df_VHVV_field.loc[:,df_VHVV_field.columns.isin(['VH_VV_' + id])][df_diff_harvest_vhvv.Date > pd.to_timedelta(0)].iloc[0:values_after_harvest])  ### Extraction of VHVV for two closest dates after harvest

            df_harvest_values_field_vhvv = df_harvest_values_field_vhvv.rename(columns={'{}'.format(id): '{}_VHVV'.format(id)})

            if df_harvest_values_field_vhvv.isnull().values.any() or df_harvest_values_field_vhvv.isna().values.any():  # avoid having data gaps
                continue
            ### combine all the S1S2 metrics in one df per field
            df_harvest_model_window.append(pd.concat([pd.DataFrame(df_harvest_values_field_cropsar.T.values,index=[id], columns=(['fAPAR_{}'.format(n) for n in range(1, values_after_harvest+values_before_harvest+1)])),pd.DataFrame(df_harvest_values_field_vhvv.T.values,index=[id], columns=(['ro110_161_VHVV_{}'.format(n) for n in range(1, values_after_harvest+values_before_harvest+1)]))],axis=1))  # pd.DataFrame(df_harvest_values_field_coherence.T.values, index = [id_cal_val], columns = (['ro110_coh_{}'.format(n) for n in range(1,5)])),

        return df_harvest_model_window


    def compile_non_harvest_data_extraction(df_cropsar_combined, df_VHVV_combined, fields_dataset_compiling, df_harvest_dates,Year_interest):
        df_non_harvest_data_extraction = []
        print('{} FIELDS TO COMPILE IN DATASET'.format(len(fields_dataset_compiling) - 1))
        for id in fields_dataset_compiling:
            df_harvest_date = df_harvest_dates.loc[df_harvest_dates.id == id]['Harvest_date']
            df_harvest_date = pd.to_datetime(df_harvest_date.values)
            if df_harvest_date.isnull() or df_harvest_date.year[0] != Year_interest:
                continue
            print('FIELD {}: COMPILING OF {}'.format(fields_dataset_compiling.index(id), id))
            df_cropsar_field = df_cropsar_combined.loc[:, df_cropsar_combined.columns.isin([id])]
            df_VHVV_field = df_VHVV_combined.loc[:, df_VHVV_combined.columns.isin(['VH_VV_' + id])]
            df_diff_harvest_cropsar = pd.DataFrame(df_cropsar_field.index - df_harvest_date.values[0])
            df_diff_harvest_cropsar.index = df_cropsar_field.index
            df_diff_harvest_cropsar.sort_index(ascending=True, inplace=True)
            df_diff_harvest_vhvv = pd.DataFrame(df_VHVV_field.index - df_harvest_date.values[0])
            df_diff_harvest_vhvv.index = df_VHVV_field.index
            df_diff_harvest_vhvv.sort_index(ascending=True, inplace=True)

            #### part to extract NON-harvest data
            ## 3 periods after harvest extraction
            counter_after_harvest_select = 0
            random_sample_excl = []
            date_range_after_harvest = pd.date_range(start=(df_VHVV_field[df_diff_harvest_vhvv.Date > pd.to_timedelta(0)].dropna().iloc[values_after_harvest+1:].index[0].strftime('%Y-%m-%d')),
                                                     end=(df_VHVV_field[df_diff_harvest_vhvv.Date > pd.to_timedelta(0)].dropna().iloc[-1:].index[0].strftime('%Y-%m-%d')), freq='6D')
            random_sample = [0]
            ## pick random date out of date range
            while random_sample and counter_after_harvest_select < 3:
                try:
                    random_sample = choice([i for i in range(1, len(date_range_after_harvest) - (values_before_harvest+values_after_harvest-1)) if i not in random_sample_excl])  # see that the boundary of the range are excluded for period selection
                except:
                    random_sample = []  # no sampling possibilities left
                if not random_sample:
                    continue
                df_no_harvest_values_field_vhvv = df_VHVV_field[df_VHVV_field.index >= date_range_after_harvest[random_sample - 1]].iloc[0:values_before_harvest+values_after_harvest]  # take all the harvest values around the selected date
                if df_no_harvest_values_field_vhvv.isnull().values.any() or df_no_harvest_values_field_vhvv.isna().values.any():
                    random_sample_excl.extend([random_sample-4, random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3, random_sample+4])
                    continue
                # df_no_harvest_values_field_coherence = df_coherence_fields_selected.loc[:,df_coherence_fields_selected.columns.isin([id_cal_val])][df_coherence_fields_selected.index >= date_range_after_harvest[random_sample-1]].iloc[0:4]
                # if df_no_harvest_values_field_coherence.isnull().values.any() or df_no_harvest_values_field_coherence.isna().values.any():
                #     random_sample_excl.extend([random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3])
                #     continue
                df_no_harvest_values_field_cropsar = df_cropsar_field[df_cropsar_field.index >= date_range_after_harvest[random_sample - 1]].iloc[0:values_before_harvest + values_after_harvest]

                df_non_harvest_data_extraction.append(pd.concat([pd.DataFrame(df_no_harvest_values_field_cropsar.T.values,index=[id + '_after_{}'.format(str(counter_after_harvest_select))],
                                                                              columns=(['fAPAR_{}'.format(n) for n in range(1, values_before_harvest+values_after_harvest+1)])),pd.DataFrame(df_no_harvest_values_field_vhvv.T.values,index=[id + '_after_{}'.format(str(counter_after_harvest_select))],
                                                                              columns=(['ro110_161_VHVV_{}'.format(n) for n in range(1, values_after_harvest + values_before_harvest+1)]))],axis=1))  # pd.DataFrame(df_no_harvest_values_field_coherence.T.values,index=[id_cal_val+'_after_{}'.format(str(counter_after_harvest_select))], columns=(['ro110_coh_{}'.format(n) for n in range(1, 5)])),
                random_sample_excl.extend([random_sample-4, random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3, random_sample +4])
                counter_after_harvest_select += 1

            ## 3 periods before harvest extraction
            counter_before_harvest_select = 0
            random_sample_excl = []
            date_range_before_harvest = pd.date_range(start=(
                df_VHVV_field[df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].dropna().iloc[:-values_before_harvest].index[0].strftime('%Y-%m-%d')),
                                                      end=(df_VHVV_field[
                                                               df_diff_harvest_vhvv.Date <= pd.to_timedelta(
                                                                   0)].dropna().iloc[:-values_before_harvest].index[-1].strftime('%Y-%m-%d')),freq='6D')
            if df_VHVV_field[
                   df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].dropna().iloc[:-values_before_harvest].index[0].month < 5:
                date_range_before_harvest = pd.date_range(start='{}-05-01'.format(str(df_harvest_date.year.values[0])),
                                                          end=(df_VHVV_field[df_diff_harvest_vhvv.Date <= pd.to_timedelta(0)].dropna().iloc[:-values_before_harvest].index[-1].strftime('%Y-%m-%d')),
                                                          freq='6D')
            random_sample = [0]
            while random_sample and counter_before_harvest_select < 3:
                try:
                    random_sample = choice([i for i in range(1, len(date_range_before_harvest) - (values_before_harvest+values_after_harvest-1)) if
                                            i not in random_sample_excl])  # see that the boundary of the range are excluded for period selection
                except:
                    random_sample = []  # no sampling possibilities left
                if not random_sample:
                    continue
                df_no_harvest_values_field_vhvv = \
                df_VHVV_field[df_VHVV_field.index >= date_range_before_harvest[random_sample - 1]].iloc[0:values_before_harvest+values_after_harvest]  # take all the harvest values around the selected date
                if df_no_harvest_values_field_vhvv.isnull().values.any() or df_no_harvest_values_field_vhvv.isna().values.any():
                    random_sample_excl.extend([random_sample-4, random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3, random_sample +4])
                    continue

                df_no_harvest_values_field_cropsar =df_cropsar_field[df_cropsar_field.index >= date_range_before_harvest[random_sample - 1]].iloc[0:values_after_harvest + values_before_harvest]

                df_non_harvest_data_extraction.append(pd.concat([pd.DataFrame(df_no_harvest_values_field_cropsar.T.values,index=[id + '_before_{}'.format(str(counter_before_harvest_select))],
                                                                              columns=(['fAPAR_{}'.format(n) for n in range(1, values_after_harvest+values_before_harvest+1)])),pd.DataFrame(df_no_harvest_values_field_vhvv.T.values,index=[id + '_before_{}'.format(str(counter_before_harvest_select))],
                                                                              columns=(['ro110_161_VHVV_{}'.format(n) for n in range(1, values_after_harvest + values_before_harvest +1)]))],
                                                                axis=1))  # pd.DataFrame(df_no_harvest_values_field_coherence.T.values,index=[id_cal_val + '_before_{}'.format(str(counter_before_harvest_select))], columns=(['ro110_coh_{}'.format(n) for n in range(1, 5)])),
                random_sample_excl.extend([random_sample -4, random_sample - 3, random_sample - 2, random_sample - 1, random_sample, random_sample + 1,random_sample + 2, random_sample + 3, random_sample +4])
                counter_before_harvest_select += 1
        return df_non_harvest_data_extraction

    df_test_harvest.extend(compile_harvest_model_data(df_cropsar_combined, df_VHVV_combined, fields_dataset_compiling,df_harvest_dates,Year_interest))
    df_test_non_harvest.extend(compile_non_harvest_data_extraction(df_cropsar_combined,df_VHVV_combined, fields_dataset_compiling, df_harvest_dates, Year_interest))

df_harvest = pd.concat(df_test_harvest)
df_non_harvest = pd.concat(df_test_non_harvest)
y_harvest = [1]*df_harvest.index.size ### columns indicatin that the extracted data is at the period of harvest
y_no_harvest = [0]*df_non_harvest.index.size ### columns indicating that the extracted data is at the period of no harvest
df_harvest['y'] = y_harvest
df_non_harvest['y'] = y_no_harvest
df_harvest_model = pd.concat([df_harvest,df_non_harvest])
df_harvest_model.index.name = 'ID_field'
if metrics_exclude:
    for s in range(len(metrics_exclude)):
        df_harvest_model.drop([col for col in df_harvest_model if metrics_exclude[s] in col],
                                        axis=1, inplace=True)
df_harvest_model.to_csv(os.path.join(outdir,'df_test_VHVV_fAPAR.csv'))

