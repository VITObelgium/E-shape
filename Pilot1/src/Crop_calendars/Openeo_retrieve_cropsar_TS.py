import json
from openeo.rest.conversions import timeseries_json_to_pandas
from cropsar.preprocessing.retrieve_timeseries_openeo import run_cropsar_dataframes
import pandas as pd
with open(r'C:\Users\bontek\git\e-shape\Pilot1\Tests\Cropcalendars\EX_files\datacube_metrics_sigma_V2.json') as json_file:
    ts_json = json.load(json_file)
    ts_df = timeseries_json_to_pandas(ts_json)
    ts_df.index = pd.to_datetime(ts_df.index).date

metrics_order =  ['gamma_VH', 'gamma_VV', 'sigma_ascending_VH', 'sigma_ascending_VV','sigma_angle','sigma_descending_VH', 'sigma_descending_VV','sigma_descending_angle', 'fAPAR']  # The index position of the metrics returned from the OpenEO datacube

def get_cropsar_TS(ts_df, Spark = False):
    index_fAPAR = metrics_order.index('fAPAR')
    df_S2 = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([str(index_fAPAR)])].sort_index().T
    df_S2 *= 0.005
    index_S1_ascending = metrics_order.index('sigma_ascending_VH')
    df_S1_ascending = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([str(index_S1_ascending), str(index_S1_ascending+1), str(index_S1_ascending +2)])].sort_index().T
    index_S1_descending = metrics_order.index('sigma_descending_VH')
    df_S1_descending = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([str(index_S1_descending), str(index_S1_descending+1), str(index_S1_descending +2)])].sort_index().T
    if Spark:
        cropsar_df, cropsar_df_q10, cropsar_df_q90 = run_cropsar_dataframes(df_S2, df_S1_ascending, df_S1_descending)
    else:
        cropsar_df = pd.read_csv(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\cropsar_df_openeo_output.csv")
        cropsar_df = cropsar_df.set_index(cropsar_df.columns[0])
    return cropsar_df
