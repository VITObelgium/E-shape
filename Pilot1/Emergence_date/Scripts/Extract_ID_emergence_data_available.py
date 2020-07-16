import geopandas as gpd
import pandas as pd
import numpy as np
crop_calendar_param = r'planting_d'
dict_meta_files = {'WIG_2018': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates_overview_reduc.xlsx",
                   'WIG_2019': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates_overview_reduc.xlsx",
                   'Flax_2018' : r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all_overview.xlsx"}
def check_crop_calendar_data_availability(dict_meta_files,calendar_parameter):
    dict_ids_available = {}
    for key in  dict_meta_files.keys():
        df_meta = pd.read_excel(dict_meta_files.get(key))
        df_meta = df_meta[[col for col in df_meta.columns if calendar_parameter in col] +['id']]
        if df_meta.shape[1] <2:
            continue
        df_meta.dropna(inplace = True)
        dict_ids_available.update({'{}'.format(key): df_meta.id.to_list()})
    return dict_ids_available



IDs_available = check_crop_calendar_data_availability(dict_meta_files, crop_calendar_param)