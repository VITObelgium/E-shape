import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os
import glob
ro_select = r'ro161'
year = 2019
start = '{}-01-01'.format(str(year))
end = '{}-12-31'.format(str(year))
idx = pd.date_range(start,end)
id_dataset = r'TAP_Monitoring_fields'
Cropsar_data = r'S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\CropSAR'
idx = pd.date_range('{}-01-01'.format(str(year)),'{}-12-31'.format(str(year)))
##### loading of df's
# df_fAPAR = pd.read_csv(r"S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\CropSAR\{}_fAPAR_{}_Flax_fields_mowing_dates_allfields.csv".format(str(year),str(year)))
# df_fAPAR = df_fAPAR.rename(columns = {'Unnamed: 0':'Date'})
# ids = list(df_fAPAR.columns)
#ids.remove('Date')
# ids = ids[1:]
# df_coherence = pd.read_csv(r"S:\eshape\Pilot 1\results\Harvest_date\S1_coherence_Flax_fields_{}_mowing_dates_{}.csv".format(str(year),ro_select))
# coh_vv_ids = np.arange(0,len(ids),1)
# coh_vh_ids = np.arange(0.1,len(ids),1)
# df_coherence = df_coherence.rename(columns = {'polygon':'Date'})
# df_coherence = df_coherence.iloc[2:]
# df_coherence.index = pd.to_datetime(df_coherence['Date'])
# df_coherence = df_coherence.drop(columns = ['Date'])
# S1_vv_ids = np.arange(1,len(ids)*2-1+0.001,2, dtype = int)
# S1_vh_ids = np.arange(0,len(ids)*2-1+0.001,2, dtype = int)
try:
    df_S1_ratio = pd.read_csv(r"S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\S1_Ascending_{}_{}_{}_{}.csv".format(str(year),str(year),id_dataset,ro_select))
except:
    df_S1_ratio = pd.read_csv(r"S:\eshape\Pilot 1\results\Harvest_date\S1_S2_data\S1_Descending_{}_{}_{}_{}.csv".format(str(year),str(year),id_dataset,ro_select))

df_S1_ratio.index = pd.to_datetime(df_S1_ratio['Date'])
df_S1_ratio = df_S1_ratio.drop(columns=['Date'])
overwrite = True
df_harvest = pd.read_csv(r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.csv")
 #df_harvest = gpd.read_file(r"S:\eshape\Pilot 1\results\Harvest_date\Training_Val_selection\2019_WIG_fields_selected.shp")
ids =df_harvest.id.to_list()

##### CROPSAR data extraction
#files_cropsar = glob.glob(os.path.join(Cropsar_data, '*parcel_{}*_cropsar.csv'.format(id_dataset)))
files_cropsar =  glob.glob(os.path.join(Cropsar_data, '*parcel_*cropsar.csv'))
df_fapar = []
for file_cropsar in files_cropsar:
    df_fapar_field = pd.read_csv(file_cropsar)
    df_fapar_field = df_fapar_field.rename(columns={'Unnamed: 0': 'Date'})
    df_fapar_field.Date = df_fapar_field.Date.values.astype('datetime64[D]')
    df_fapar_field.index = df_fapar_field.Date
    df_fapar_field = df_fapar_field.drop(columns=['Date'])
    df_fapar_field = pd.DataFrame(df_fapar_field['q50'])
    df_fapar_field = df_fapar_field.rename(columns={'q50': '{}'.format(os.path.split(file_cropsar)[-1].split('parcel_')[-1].split('_cropsar')[0].replace('{}_'.format(id_dataset),''))})
    df_fapar.append(df_fapar_field)
df_fapar = pd.concat(df_fapar, axis=1)
df_fapar = df_fapar.reindex(idx)

for id in ids:
    df_harvest_date = df_harvest.loc[df_harvest.id == id]['harvest_da']
    df_harvest_date = pd.to_datetime(df_harvest_date.values)
    crop_type = df_harvest.loc[df_harvest.id == id]['croptype'].values[0]



    if not pd.isnull(df_harvest_date[0]):
        ### plotting of fAPAR data per field
        fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 10))
    else:
        continue

    df_id_fapar  = df_fapar.loc[:,df_fapar.columns.isin([id])]
    df_id_fapar.plot(grid = True, ax = ax1, color = 'green', label = 'fAPAR_smoothed')
    #ax1.set_ylim(bottom = 0, top = 1)
    ax1.axvline(x=df_harvest_date[0], color='red', label='Harvest')
    ax1.set_ylabel('fAPAR')
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper left')
    ax1.set_title('fAPAR and VHVV ratio {}'.format(str(crop_type)))

    ###### VH/VV ratio extraction
    df_S1_ratio_field =  pd.concat([df_S1_ratio.iloc[:,df_S1_ratio.columns.isin(['VH_'+id])],df_S1_ratio.iloc[:,df_S1_ratio.columns.isin(['VV_'+id])]],axis = 1)
    df_S1_ratio_field['VH_VV_ratio'] = 10*np.log(df_S1_ratio_field['VH_'+id]/df_S1_ratio_field['VV_'+id])

    df_S1_ratio_field['VH_VV_ratio'].plot(grid=True, ax=ax2, color='blue', label='VH_VV_ratio_{}_dB'.format(ro_select))
    ax2.axvline(x=df_harvest_date[0], color='red', label='Harvest')
    ax2.set_ylabel('VH/VV ratio')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    plt.tight_layout()
    if not os.path.exists(os.path.join(r'S:\eshape\Pilot 1\results\Harvest_date\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type)),'{}_{}_{}_fAPAR_cropSAR_Coherence_S1_VH_VV_{}.png'.format(str(year),str(id),id_dataset,ro_select))) or overwrite:
        if not os.path.exists(os.path.join(r'S:\eshape\Pilot 1\results\Harvest_date\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type)))): os.makedirs(os.path.join(r'S:\eshape\Pilot 1\results\Harvest_date\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type))))
        fig.savefig(os.path.join(r'S:\eshape\Pilot 1\results\Harvest_date\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type)),'{}_{}_{}_fAPAR_cropSAR_Coherence_S1_VH_VV_{}.png'.format(str(year),str(id),id_dataset,ro_select)))

        plt.close()


