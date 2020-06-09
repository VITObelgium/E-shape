import pandas as pd
import glob
import os
import geopandas as gpd
import numpy as np
from datetime import datetime

start = '2018-01-01'
end = '2018-12-31'
idx = pd.date_range(start,end)
WIG_fields = gpd.read_file(r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all.shp")
ids = WIG_fields.id.to_list()
df_combine = pd.DataFrame()
# df_combine['Date']= pd.date_range(start,end,freq='D')
# df_combine = df_combine.set_index('Date')

##### fAPAR
csv_files = glob.glob(os.path.join(r'S:\eshape\Pilot 1\results\tmp','2018_fAPAR_Flax*.CSV'))

for id in ids:
    csv_select = [item for item in csv_files if id in item]
    df = pd.read_csv(os.path.join(r'S:\eshape\Pilot 1\results',csv_select[0]), header = None)
    df.index = pd.to_datetime(df[0])
    df = df.reindex(idx,fill_value = np.nan)
    df_combine['{}'.format(str(id))] = df[1]
if not os.path.exists(os.path.join(r'S:\eshape\Pilot 1\results','2018_fAPAR_2018_Flax_fields_mowing_dates_allfields.csv')):
    df_combine.to_csv(os.path.join(r'S:\eshape\Pilot 1\results','2018_fAPAR_2018_Flax_fields_mowing_dates_allfields.csv'))


### S1 data
# csv_files = glob.glob(os.path.join(r'S:\eshape\Pilot 1\results\tmp','2018_S1_Ascending_Flax*.CSV'))
# t_88 = pd.date_range("2018-01-05","2018-12-31",freq="6D",tz = 'utc').to_pydatetime()
# df_combine['Date']= t_88
# df_combine = df_combine.set_index('Date')
# df_combine['Date'] = [item.strftime('%Y-%m-%d') for item in list(df_combine.index)]
# for id in ids:
#     csv_select = [item for item in csv_files if id in item]
#     df = pd.read_csv(os.path.join(r'S:\eshape\Pilot 1\results',csv_select[0]), header = None)
#     df = df.iloc[1:]
#     df.index = pd.to_datetime(df[0], utc = True)
#     #df = df.reindex(idx, fill_value=np.nan)
#     #df = df.reindex(t_88,fill_value = np.nan)
#     try:
#         df = df.loc[t_88]
#         df = df.drop(columns=[0])
#         df.columns = ['VH_{}'.format(id), 'VV_{}'.format(id), 'angle_{}'.format(id)]
#     except:
#         df = pd.DataFrame()
#         df['Date'] = t_88
#         df = df.set_index('Date')
#         df['0'] = [np.nan]*len(df.index)
#         df['1'] = [np.nan]*len(df.index)
#     df_combine[['VH_{}'.format(id),'VV_{}'.format(id)]] = df.iloc[:,0:2]
# if not os.path.exists(os.path.join(r'S:\eshape\Pilot 1\results','S1_Ascending_2019_2019_Flax_fields_planting_harvest_dates_ro88.csv')):
#     df_combine.to_csv(os.path.join(r'S:\eshape\Pilot 1\results','S1_Ascending_2019_2019_Flax_fields_planting_harvest_dates_ro88.csv'), index= False)


