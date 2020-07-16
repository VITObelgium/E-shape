import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os
import  glob


year = 2018
df_harvest = pd.read_excel(r"S:\eshape\Pilot 1\data\WIG_data\{}_WIG_planting_harvest_dates_overview.xlsx".format(str(year)), sheet_name='{}_WIG_planting_harvest_dates'.format(str(year)))
ros = ['ro88','ro110','ro161']
VI_data = r'S:\eshape\Pilot 1\results\Harvest_date\tmp'
IDs_harvest_date = r'S:\eshape\Pilot 1\results\Harvest_date\plots'
ids_fapar = []
counter = 0

#### See for which field IDs some data on harvest date, fAPAR and coherence is available
for ro in ros:
    folder =os.path.join(IDs_harvest_date,ro,str(year))
    ids= glob.glob(folder+r'\\**\\*Coherence_{}.png'.format(ro))
    ids = [os.path.split(item)[-1].rsplit('_fAPAR')[0].rsplit(str(year)+'_')[1] for item in ids]
    ids_fapar.append(ids)
    if counter == 0:
        ids_orbit = {ro: ids}
    else:
        ids_orbit_tmp = {ro: ids}
        ids_orbit.update(ids_orbit_tmp)
    counter +=1
ids_fapar = [item for sublist in ids_fapar for item in sublist] ### all the ids that can  be used to built a training/validation dataset
ids_fapar = list(set(ids_fapar))
df_fAPAR = pd.read_csv(r"S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\{}_fAPAR_{}_WIG_planting_harvest_dates_allfields.csv".format(str(year),str(year)))
ids_fields = list(df_fAPAR.columns) ### all field IDS (reference) with some data on fAPAR for tht year
ids_fields.remove('Date')
vh_ids = np.arange(0.1,len(ids_fields),1) # the name of the IDS in the coherence dataframe based on the amount of reference fields in the fAPAR dataframe
vh_ids = [str(item) for item in vh_ids]
ro_loop_counter = 0

#### In the loop below some coherence and fAPAR data will be extracted (around the harvest date) for all fields with available harvest data in the year of interest, data from 3 RO and fAPAR will be used for these IDs
ids_remove_no_coherence = []
for ro in ros:
    print('\n extracting data for RO: {}'.format(ro))
    counter_ids_loop = 0
    df_coherence = pd.read_csv(r"S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\S1_coherence_{}_{}_WIG_planting_harvest_dates_{}.csv".format(str(year), str(year), ro))
    df_coherence = df_coherence.rename(columns={'polygon': 'Date'})
    df_coherence = df_coherence.iloc[2:]
    dates_coherence = df_coherence.Date.to_list()
    dates_coherence = [item.rsplit(' ')[0] for item in dates_coherence]
    df_coherence['Date'] = pd.to_datetime(dates_coherence)
    df_coherence = df_coherence.drop(columns=list(vh_ids)) ### only use the VV coherence (more sensitive)
    columns_names = ids_fields
    if ro_loop_counter == 0 and counter_ids_loop == 0:
        columns_names.insert(0,'Date')
    df_coherence.columns =columns_names
    for id in ids_fapar:
        coherence_harvest_model  = []
        print('extracting data for field: {}'.format(id))
        df_harvest_date = df_harvest.loc[df_harvest.id == id]['harvest_da']
        df_harvest_date = pd.to_datetime(df_harvest_date.values)
        if not df_harvest_date.year.values[0] == year:
            continue
        if ro_loop_counter == 0:
            df_fAPAR_extract = df_fAPAR[['Date', id]] # extract the id of interest
            df_fAPAR_extract = df_fAPAR_extract.dropna()
            df_fAPAR_extract['Date'] = pd.to_datetime(df_fAPAR_extract.Date)

            df_fAPAR_extract['diff_harvest'] = (df_fAPAR_extract.Date-df_harvest_date.values[0])
            df_fAPAR_extract.sort_values(by=['diff_harvest'], ascending=True, inplace=True)
            fAPAR_harvest_model = [df_fAPAR_extract[df_fAPAR_extract.diff_harvest < pd.to_timedelta(0)].iloc[-1][id]] #### Addition of fAPAR for closest date before harvest
            fAPAR_harvest_model.extend(df_fAPAR_extract[df_fAPAR_extract.diff_harvest > pd.to_timedelta(0)].iloc[0:2][id].to_list()) ### Extraction of fAPAR for two closest dates after harvest
            fAPAR_harvest_model = [item*0.005 for item in fAPAR_harvest_model]
            if len(fAPAR_harvest_model) <3:
                fAPAR_harvest_model.append(np.nan)

            if counter_ids_loop == 0:
                fAPAR_dict = {'{}_fAPAR'.format(id):fAPAR_harvest_model}
            else:
                fAPAR_dict_tmp =  {'{}_fAPAR'.format(id):fAPAR_harvest_model}
                fAPAR_dict.update(fAPAR_dict_tmp)

        #### coherence extraction for the specific id and ro
        df_coherence_extract = df_coherence[['Date',id]]
        df_coherence_extract = df_coherence_extract.dropna()
        if df_coherence_extract.empty:
            ids_remove_no_coherence.append(id)
            continue
        df_coherence_extract['diff_harvest'] = (df_coherence_extract.Date-df_harvest_date.values[0])
        df_coherence_extract.sort_values(by = ['diff_harvest'], ascending = True, inplace=True)
        coherence_harvest_model.append(df_coherence_extract[df_coherence_extract.diff_harvest< pd.to_timedelta(0)].iloc[-2:][id].to_list()) # before
        coherence_harvest_model.append(df_coherence_extract[df_coherence_extract.diff_harvest > pd.to_timedelta(0)].iloc[0:2][id].to_list())  # after
        coherence_harvest_model = [item for sublist in coherence_harvest_model for item in sublist]
        coherence_harvest_model = [item*0.004 for item in coherence_harvest_model ]
        if counter_ids_loop == 0 and ro_loop_counter == 0:
            coherence_dict = {'{}_{}_Coherence'.format(id,ro): coherence_harvest_model}
        else:
            coherence_dict_tmp = {'{}_{}_Coherence'.format(id,ro): coherence_harvest_model}
            coherence_dict.update(coherence_dict_tmp)

        counter_ids_loop += 1

    ro_loop_counter += 1

########### Building a validation dataframe which nicely show all the validation data per field ID
ids_validation = list(fAPAR_dict.keys())
ids_remove_no_coherence  = list(set(ids_remove_no_coherence))
ids_validation = [item.rsplit('_fAPAR')[0] for item in ids_validation]
ids_validation = [item for item in ids_validation if item not in ids_remove_no_coherence]
df_validation = pd.DataFrame({'IDs': ids_validation})
column_id_ro88 = ['ro88_coh_{}'.format(n) for n in range(1,5)]
column_id_ro110 = ['ro110_coh_{}'.format(n) for n in range(1,5)]
column_id_ro161 = ['ro161_coh_{}'.format(n) for n in range(1,5)]
column_id_fAPAR = ['fAPAR_{}'.format(n) for n in range(1,4)]
columns_df = column_id_ro88+column_id_ro110+column_id_ro161+column_id_fAPAR
columns_df.insert(0,'IDs')
counter = 0
for id in ids_validation:
    ro_88_data = coherence_dict.get('{}_ro88_Coherence'.format(id))
    ro_110_data = coherence_dict.get('{}_ro110_Coherence'.format(id))
    ro_161_data = coherence_dict.get('{}_ro161_Coherence'.format(id))
    fAPAR_data = fAPAR_dict.get('{}_fAPAR'.format(id))
    validation_data_df_tmp = pd.concat(
    [pd.DataFrame(ro_88_data, index=[item for item in columns_df if 'ro88' in item], columns=[id]),
     pd.DataFrame(ro_110_data, index=[item for item in columns_df if 'ro110' in item], columns=[id]),
     pd.DataFrame(ro_161_data, index=[item for item in columns_df if 'ro161' in item], columns=[id]),
     pd.DataFrame(fAPAR_data, index=[item for item in columns_df if 'fAPAR' in item], columns=[id])], axis=0,sort=False)

    if counter  == 0:
        validation_data_df = validation_data_df_tmp
    if counter >0:
       validation_data_df= pd.concat([validation_data_df,validation_data_df_tmp])
    counter += 1



df_validation = validation_data_df.max(level =0)
df_validation = df_validation.T
y = [1]*df_validation.index.size ### columns indicatin that the extracted data is at the period of harvest
df_validation['y'] = y
df_validation.index.name = 'ID_field'


df_validation.to_csv(r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Mehrdad\validation\Validation_data_eshape_{}_update_order.csv'.format(str(year)))















