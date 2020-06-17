import openeo
from openeo.rest.conversions import timeseries_json_to_pandas
import pandas as pd
import geopandas as gpd
#import geojson
import shapely
from shapely.ops import transform
from functools import partial
import pyproj
import numpy as np
import os
import scipy.signal
import json
def create_mask(start, end, session):
    s2_sceneclassification = session.imagecollection("S2_FAPAR_SCENECLASSIFICATION_V102_PYRAMID",
                                                     bands=["classification"])

    classification = s2_sceneclassification.band('classification')

    def makekernel(iwindowsize):
        kernel_vect = scipy.signal.windows.gaussian(iwindowsize, std=iwindowsize / 6.0, sym=True)
        kernel = np.outer(kernel_vect, kernel_vect)
        kernel = kernel / kernel.sum()
        return kernel

    # in openEO, 1 means mask (remove pixel) 0 means keep pixel

    # keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17)) # make small kernel for buffering around pixels whihc belongs not to the suitable classes
    # remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    first_mask = first_mask > 0.057

    # remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10)
    second_mask = second_mask.apply_kernel(makekernel(161)) # bigger kernel for cloud pixels to remove from a larger area pixels
    second_mask = second_mask > 0.1

    # TODO: the use of filter_temporal is a trick to make cube merging work, needs to be fixed in openeo client
    return first_mask.filter_temporal(start, end) | second_mask.filter_temporal(start, end)

def get_geometry(field, region='Flanders'):
    ###################### DEFINE GEOMETRY OF POLGYON #######################
    #########################################################################
    geometry = field['geometry']
    geometry = shapely.ops.transform(lambda x, y, z=0: (x, y), geometry)
    if region == 'Wallonia':
        project = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:31370'))
        geometry = transform(project, geometry)
        field['geometry'] = geometry
    field['geometry'] = geometry
    geometry = geojson.Feature(geometry=geometry).geometry
    bbox = field.geometry
    minx, miny, maxx, maxy = bbox.bounds
    bbox = bbox.bounds
    return bbox, geometry, minx, miny, maxx, maxy, field

shp_dir = r"S:\eshape\Pilot 1\data\WIG_data\Field_error.shp"
WIG_fields = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\Field_error.shp")
start = '2019-01-01'
end = '2019-12-31'
start_year = start[0:4]
end_year = end[0:4]
start_month = start[5:7]
end_month = end[5:7]
start_day = start[8:10]
end_day = end[8:10]
#connection_coherence = openeo.connect("http://openeo-dev.vgt.vito.be/openeo/0.4.0")
connection = openeo.connect("http://openeo.vgt.vito.be/openeo/0.4.0")
connection.authenticate_basic("bontek","bontek123")
outdir = r'S:\eshape\Pilot 1\results\tmp'
#for index,field in WIG_fields.iterrows():
#
#fAPAR
# for index, field in WIG_fields.iterrows():
#     if not os.path.exists("{}.csv".format(os.path.join(outdir,'tmp',start_year+'_'+'fAPAR'+'_Flax_fields_'+field.id))):
#         datacube = connection.load_collection("S2_FAPAR_V102_WEBMERCATOR2")
#         mask = create_mask(start,end,connection)
#         s1_time_series_dict = datacube \
#             .filter_temporal(start, end) \
#             .mask(rastermask= mask, replacement= np.nan)\
#             .polygonal_mean_timeseries(field.geometry) \
#             .execute()
#
#         ids = list(WIG_fields.id.values)
#         df_FAPAR = timeseries_json_to_pandas(s1_time_series_dict)
#         df_FAPAR.columns = ids
#         df_FAPAR.index = pd.to_datetime(df_FAPAR.index)
#         df_FAPAR.dropna(how='all', inplace=True)
#         df_FAPAR.to_csv("{}.csv".format(os.path.join(outdir,'tmp',start_year+'_'+'fAPAR'+'_Flax_fields'+field.id)))
## coherence
# datacube = connection_coherence.load_collection("TERRASCOPE_S1_SLC_COHERENCE_V1")
# s1_time_series_dict = datacube \
#     .filter_temporal(start, end) \
#     .polygonal_mean_timeseries(WIG_fields) \
#     .execute_batch(os.path.join(outdir, 'S1_coherence_'+'TAP_Monitoring_fields_' + start_year+'_'+os.path.split(shp_dir)[-1][0:-4]+'.json')) #.format(os.path.join(outdir,'S1_coherence_'+start_year)))
# for index, field in WIG_fields.iterrows():
#     if not os.path.exists("{}.csv".format(os.path.join(outdir, 'tmp', start_year + '_' + 'S1_Coherence' + '_TAP_Monitoring_fields_' + field.id))):
#         coh_time_series_dict = datacube \
#                 .filter_temporal(start, end) \
#                 .polygonal_mean_timeseries(field.geometry) \
#                 .execute()
#         df_coherence = timeseries_json_to_pandas(coh_time_series_dict)
#         df_coherence.index = pd.to_datetime(df_coherence.index)
#         df_coherence.dropna(how='all', inplace=True)
#         df_coherence.to_csv("{}.csv".format(os.path.join(outdir,'tmp',start_year+'_'+'S1_Coherence'+'_TAP_Monitoring_fields_'+field.id)))

#with open("{}.json".format(os.path.join(outdir, 'S1_coherence_'+'Flax_fields_' + start_year+'_'+os.path.split(shp_dir)[-1][0:-4])), 'r') as c:
# with open(r"S:\eshape\Pilot 1\results\S1_coherence_Flax_fields_2018_vlas_2018_wgs_all.json") as c:
#     ts_dict_c = json.load(c)
#     t_88 = pd.date_range("2018-01-05","2018-12-31",freq="6D",tz = 'utc').to_pydatetime()
#     t_88 = t_88.tolist()
#     df_Coherence = timeseries_json_to_pandas(ts_dict_c)
#     df_Coherence.index = pd.to_datetime(df_Coherence.index)
#     df_Coherence = df_Coherence.loc[t_88]
#     df_Coherence.dropna(how='all', inplace=True)
#     df_Coherence.to_csv("{}.csv".format(os.path.join(outdir, 'S1_coherence_' +'Flax_fields_'+ start_year+'_'+os.path.split(shp_dir)[-1][0:-4]+'_ro88')))
#


############## S1 VV/VH data extraction
#datacube = connection.load_collection('S1_GRD_SIGMA0_ASCENDING')
# s1_time_series_ascending = datacube\
#         .filter_temporal(start,end)\
#     .polygonal_mean_timeseries(WIG_fields)\
#     .execute_batch(os.path.join(outdir, 'S1_Ascending_'+'TAP_Monitoring_fields_' + start_year+'_'+os.path.split(shp_dir)[-1][0:-4]+'.json'))


# for index, field in WIG_fields.iterrows():
#     if not os.path.exists("{}.csv".format(os.path.join(outdir,'tmp',start_year+'_'+'S1_Ascending'+'_TAP_Monitoring_fields_'+field.id))):
#         s1_time_series_ascending = datacube \
#             .filter_temporal(start, end) \
#             .polygonal_mean_timeseries(field.geometry) \
#             .execute()
#
#         ids = list(WIG_fields.id.values)
#         df_S1 = timeseries_json_to_pandas(s1_time_series_ascending)
#         #df_S1.columns = ids
#         df_S1.index = pd.to_datetime(df_S1.index)
#         df_S1.dropna(how='all', inplace=True)
#         df_S1.to_csv("{}.csv".format(os.path.join(outdir,'tmp',start_year+'_'+'S1_Ascending'+'_TAP_Monitoring_fields_'+field.id)))

datacube2 =  connection.load_collection('S1_GRD_SIGMA0_DESCENDING')
# s1_time_series_descending = datacube2\
#         .filter_temporal(start,end)\
#     .polygonal_mean_timeseries(WIG_fields)\
#     .execute_batch(os.path.join(outdir, 'S1_Descending_' + start_year+'_'+os.path.split(shp_dir)[-1][0:-4]+'.json'))

#
for index, field in WIG_fields.iterrows():
    if not os.path.exists("{}.csv".format(os.path.join(outdir,start_year+'_'+'S1_Descending'+'_TAP_Monitoring_fields_' +field.id+'_Test'))):
        s1_time_series_descending = datacube2 \
            .filter_temporal(start, end) \
            .polygonal_mean_timeseries(field.geometry) \
            .execute()

        ids = list(WIG_fields.id.values)
        df_S1 = timeseries_json_to_pandas(s1_time_series_descending)
        #df_S1.columns = ids
        df_S1.index = pd.to_datetime(df_S1.index)
        df_S1.dropna(how='all', inplace=True)
        df_S1.to_csv("{}.csv".format(os.path.join(outdir,start_year+'_'+'S1_Descending'+'_TAP_Monitoring_fields_'+field.id+'_Test')))










##### other method for getting job from openeo
        # job = datacube \
        #     .filter_temporal(start, end) \
        #     .polygonal_mean_timeseries(field.geometry)\
        #     .send_job()
        # job_id = job.job_id
        # job = RESTJob(job_id,connection)
        # job.download_results("{}.json".format(os.path.join(outdir,'S1_coherence_'+start_year+'_'+field.id)))






