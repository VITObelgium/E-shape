import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os
import glob
ro_select = r'ro59'
year = 2018
shp = gpd.read_file(r"S:\eshape\Pilot 1\data\Flax_fields\vlas_{}_wgs_all.shp".format(str(year)))
start = '{}-01-01'.format(str(year))
end = '{}-12-31'.format(str(year))
idx = pd.date_range(start,end)
##### loading of df's
df_fAPAR = pd.read_csv(r"S:\eshape\Pilot 1\results\S1_S2_data\{}_fAPAR_{}_Flax_fields_mowing_dates_allfields.csv".format(str(year),str(year)))
df_fAPAR = df_fAPAR.rename(columns = {'Unnamed: 0':'Date'})
ids = list(df_fAPAR.columns)
#ids.remove('Date')
ids = ids[1:]
df_coherence = pd.read_csv(r"S:\eshape\Pilot 1\results\S1_coherence_Flax_fields_{}_mowing_dates_{}.csv".format(str(year),ro_select))
coh_vv_ids = np.arange(0,len(ids),1)
coh_vh_ids = np.arange(0.1,len(ids),1)
df_coherence = df_coherence.rename(columns = {'polygon':'Date'})
df_coherence = df_coherence.iloc[2:]
df_coherence.index = pd.to_datetime(df_coherence['Date'])
df_coherence = df_coherence.drop(columns = ['Date'])
S1_vv_ids = np.arange(1,len(ids)*2-1+0.001,2, dtype = int)
S1_vh_ids = np.arange(0,len(ids)*2-1+0.001,2, dtype = int)
try:
    df_S1_ratio = pd.read_csv(r"S:\eshape\Pilot 1\results\S1_Ascending_{}_{}_Flax_fields_mowing_dates_{}.csv".format(str(year),str(year),ro_select))
except:
    df_S1_ratio = pd.read_csv(r"S:\eshape\Pilot 1\results\S1_Descending_{}_{}_Flax_fields_mowing_dates_{}.csv".format(str(year),str(year),ro_select))

df_S1_ratio.index = pd.to_datetime(df_S1_ratio['Date'])
df_S1_ratio = df_S1_ratio.drop(columns=['Date'])
overwrite = True
for p in range(len(ids)):
    df_event1_field = shp.loc[shp.id == ids[p]]['Tijdstip v']
    df_event1_field = pd.to_datetime(df_event1_field.values,dayfirst=True)
    df_event2_field = shp.loc[shp.id == ids[p]]['Tijdstip i']
    df_event2_field = pd.to_datetime(df_event2_field.values, dayfirst= True)

    ### plotting of fAPAR data per field
    crop_type = 'Flax'
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 10))


    df_id_fAPAR = pd.DataFrame({'fAPAR': df_fAPAR[ids[p]]*0.005})
    if not np.isnan(df_id_fAPAR['fAPAR']).all():
        df_id_fAPAR.index = pd.to_datetime(df_fAPAR.Date)
        df_id_fAPAR['fAPAR_smoothed'] = df_id_fAPAR.fAPAR.asfreq('D').interpolate(method = 'time')
        df_id_fAPAR['fAPAR_smoothed'].plot(grid = True, ax = ax1, color = 'green', label = 'fAPAR_smoothed')
        ax1.scatter(x = df_id_fAPAR.index,y = df_id_fAPAR.fAPAR, s = 50, c = 'black', label = 'fAPAR_raw')
        #ax1.set_ylim(bottom = 0, top = 1)
        ax1.axvline(x=df_event1_field[0], color='red', label='Tijdsstip_V')
        ax1.axvline(x=df_event2_field[0], color='red', label='Tijdsstip_I')
        ax1.set_ylabel('fAPAR')
        ax1.set_xlabel('Date')
        ax1.legend(loc='upper left')
        ax1.set_title('fAPAR and coherence for {}'.format(str(crop_type)))
    else:
        continue

    #### coherence data extraction

    df_id_coherence = pd.DataFrame({'vv_coh':df_coherence[str(coh_vv_ids[p])]*0.004,'vh_coh': df_coherence[str(coh_vh_ids[p])]*0.004})
    if not (np.isnan(df_id_coherence['vv_coh']).all()): #or np.isnan(df_id_coherence['vh_coh']).all())
        df_id_coherence['vv_coh'].plot(grid = True, ax = ax2, color = 'blue', label = 'vv_coherence_{}'.format(ro_select))
        ax2.set_ylabel('Coherence')
        ax2.set_xlabel('Date')
        #ax2.set_ylim(bottom = 0, top = 1)
        ax2.axvline(x=df_event1_field[0], color='red', label='Tijdsstip_V')
        ax2.axvline(x=df_event2_field[0], color='red', label='Tijdsstip_I')
        ax2.legend(loc = 'upper left')

        # df_id_coherence['vh_coh'].plot(grid = True, ax = ax3, color = 'blue', label = 'vh_coherence_{}'.format(ro_select))
        # #ax3.set_ylim(bottom = 0, top = 1)
        # ax3.set_ylabel('Coherence')
        # ax3.set_xlabel('Date')
        # ax3.axvline(df_harvest_date[0], color='red', label='Harvest')
        # ax3.legend(loc='upper left')
        #plt.tight_layout()

    ###### VH/VV ratio extraction
        df_S1_ratio_field =  pd.DataFrame({'S1_VH':df_S1_ratio.iloc[:,S1_vh_ids[p]],'S1_VV':df_S1_ratio.iloc[:,S1_vv_ids[p]]})
        df_S1_ratio_field['VH_VV_ratio'] = 10*np.log(df_S1_ratio_field['S1_VH']/df_S1_ratio_field['S1_VV'])
        df_S1_ratio_field['VH_VV_ratio'].plot(grid=True, ax=ax3, color='blue', label='VH_VV_ratio_{}_dB'.format(ro_select))
        ax3.axvline(x=df_event1_field[0], color='red', label='Tijdsstip_V')
        ax3.axvline(x=df_event2_field[0], color='red', label='Tijdsstip_I')
        ax3.set_ylabel('VH/VV ratio')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        plt.tight_layout()
        if not os.path.exists(os.path.join(r'S:\eshape\Pilot 1\results\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type)),'{}_{}_fAPAR_Coherence_S1_VH_VV_{}.png'.format(str(year),str(ids[p]),ro_select))) or overwrite:
            if not os.path.exists(os.path.join(r'S:\eshape\Pilot 1\results\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type)))): os.makedirs(os.path.join(r'S:\eshape\Pilot 1\results\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type))))
            fig.savefig(os.path.join(r'S:\eshape\Pilot 1\results\plots\{}\{}\{}'.format(ro_select,str(year),str(crop_type)),'{}_{}_fAPAR_Coherence_S1_VH_VV_{}.png'.format(str(year),str(ids[p]),ro_select)))

            plt.close()
    else:
        continue

