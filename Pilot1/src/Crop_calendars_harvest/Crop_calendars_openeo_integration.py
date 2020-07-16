import numpy as np
import openeo
import geopandas as gpd
import pandas as pd
import scipy.signal
import shapely
from openeo.rest.conversions import timeseries_json_to_pandas
import uuid
from datetime import timedelta
from tensorflow.keras.models import load_model


# General approach:
#
# first merge all required inputs into a single multiband raster datacube
# compute timeseries for one or more fields, for all bands (one step)
# do postprocessing of the timeseries:
#   compute cropsar based on cleaned fapar + sigma0
#   combine cropsar + coherence to determine cropcalendar
#   return cropcalendar output in your own json format

def makekernel(size: int) -> np.ndarray:
    assert size % 2 == 1
    kernel_vect = scipy.signal.windows.gaussian(size, std=size / 3.0, sym=True)
    kernel = np.outer(kernel_vect, kernel_vect)
    kernel = kernel / kernel.sum()
    return kernel


## cropsar masking function, probably still needs an update!
def create_advanced_mask(band, startdate, enddate, band_math_workaround=True):
    # in openEO, 1 means mask (remove pixel) 0 means keep pixel
    classification = band

    # keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17))
    # remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    if band_math_workaround:
        first_mask = first_mask.add_dimension("bands", "mask", type="bands").band("mask")
    first_mask = first_mask > 0.057

    # remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10)
    second_mask = second_mask.apply_kernel(makekernel(161))
    if band_math_workaround:
        second_mask = second_mask.add_dimension("bands", "mask", type="bands").band("mask")
    second_mask = second_mask > 0.1

    # TODO: the use of filter_temporal is a trick to make cube merging work, needs to be fixed in openeo client
    return first_mask.filter_temporal(startdate, enddate) | second_mask.filter_temporal(startdate, enddate)
    # return first_mask | second_mask
    # return first_mask


def get_bands(startdate,enddate):
    eoconn=openeo.connect('http://openeo-dev.vgt.vito.be/openeo/1.0.0/')
    eoconn.authenticate_basic('bontek','bontek123')

    S2mask=create_advanced_mask(eoconn.load_collection('TERRASCOPE_S2_TOC_V2',bands=['SCENECLASSIFICATION_20M']).band('SCENECLASSIFICATION_20M'),startdate=startdate,enddate=enddate)
    fapar = eoconn.load_collection('TERRASCOPE_S2_FAPAR_V2')

    fapar_masked=fapar.mask(S2mask)

    gamma0=eoconn.load_collection('TERRASCOPE_S1_GAMMA0_V1')
    coherence=eoconn.load_collection('TERRASCOPE_S1_SLC_COHERENCE_V1')

    all_bands = gamma0.merge(fapar_masked)#.merge(coherence)
    return all_bands

### ingest the field(s) for which the crop calendars should be defined
fieldgeom={
    "type":"FeatureCollection",
    "name":"small_field",
    "crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features":[
        {"type":"Feature","properties":{"id": "Sm-AVnIBAkWx2x-Z50Lg"},"geometry":{"type":"Polygon","coordinates":[[ [ 5.0877448, 51.2102439 ], [ 5.0877557, 51.2102265 ], [ 5.0877778, 51.2102198 ], [ 5.0877981, 51.2102221 ], [ 5.0885078, 51.2104537 ], [ 5.089037, 51.2106192 ], [ 5.0897114, 51.2108487 ], [ 5.0901027, 51.2109905 ], [ 5.0903599, 51.211091 ], [ 5.0904032, 51.2111066 ], [ 5.0904717, 51.2111286 ], [ 5.0905421, 51.2111483 ], [ 5.0906524, 51.2111761 ], [ 5.0908275, 51.2112134 ], [ 5.0908685, 51.2112197 ], [ 5.0909979, 51.2112374 ], [ 5.0911812, 51.2112567 ], [ 5.0912826, 51.2112633 ], [ 5.0914916, 51.2112632 ], [ 5.0915519, 51.2112616 ], [ 5.0916687, 51.2112547 ], [ 5.0918586, 51.2112402 ], [ 5.0919153, 51.2115925 ], [ 5.0919505, 51.2120637 ], [ 5.0919407, 51.2121546 ], [ 5.090201, 51.2117554 ], [ 5.0896727, 51.21163 ], [ 5.0887015, 51.2114136 ], [ 5.0881537, 51.2112869 ], [ 5.0880128, 51.2112429 ], [ 5.0879584, 51.2112134 ], [ 5.0879113, 51.2111768 ], [ 5.0878814, 51.2111483 ], [ 5.0878526, 51.2111016 ], [ 5.0878437, 51.2110722 ], [ 5.0877946, 51.2106611 ], [ 5.0877448, 51.2102439 ] ]]}},
       {"type":"Feature","properties":{"id": "SW9zVnIBAkWx2x-ZGELN"},"geometry":{"type":"Polygon","coordinates":[[[ 5.0877256, 51.2100549 ], [ 5.0877014, 51.2098494 ], [ 5.0878754, 51.2098678 ], [ 5.0879, 51.2098679 ], [ 5.0879222, 51.2098659 ], [ 5.0879283, 51.2098648 ], [ 5.0879448, 51.2098618 ], [ 5.0880239, 51.2098366 ], [ 5.0881476, 51.2098059 ], [ 5.0881721, 51.2098035 ], [ 5.0882034, 51.2098028 ], [ 5.0882118, 51.2098026 ], [ 5.0882236, 51.2098029 ], [ 5.0882584, 51.2098042 ], [ 5.0885235, 51.2098204 ], [ 5.0886116, 51.2098289 ], [ 5.0886898, 51.2098282 ], [ 5.0887098, 51.2098264 ], [ 5.0887316, 51.209823 ], [ 5.088759, 51.2098245 ], [ 5.0888208, 51.2098087 ], [ 5.0888614, 51.2098023 ], [ 5.0892029, 51.2097254 ], [ 5.0893046, 51.2096998 ], [ 5.0894006, 51.2096755 ], [ 5.0894261, 51.2096677 ], [ 5.0894458, 51.209664 ], [ 5.089504, 51.2096624 ], [ 5.089545, 51.2096639 ], [ 5.0896088, 51.2096703 ], [ 5.0896244, 51.2096718 ], [ 5.0897314, 51.2096777 ], [ 5.089808, 51.2096785 ], [ 5.0898377, 51.2096777 ], [ 5.0898686, 51.2096768 ], [ 5.089888, 51.2096738 ], [ 5.0899, 51.2096693 ], [ 5.0899014, 51.2096681 ], [ 5.0899063, 51.2096642 ], [ 5.0899056, 51.2096515 ], [ 5.0899011, 51.2096414 ], [ 5.0899551, 51.2096274 ], [ 5.0899571, 51.2096459 ], [ 5.0899589, 51.2096487 ], [ 5.0899608, 51.2096517 ], [ 5.0899684, 51.2096553 ], [ 5.0897026, 51.2105992 ], [ 5.0891133, 51.2104365 ], [ 5.0890155, 51.2105342 ], [ 5.0888922, 51.2104927 ], [ 5.0888461, 51.2104538 ], [ 5.0888184, 51.2104382 ], [ 5.0882056, 51.2102351 ], [ 5.0878607, 51.2101429 ], [ 5.0877658, 51.2101144 ], [ 5.0877433, 51.210102 ], [ 5.087734, 51.2100837 ], [ 5.0877256, 51.2100549 ] ]]}}
    ]
}
geo=shapely.geometry.GeometryCollection([shapely.geometry.shape(feature["geometry"]).buffer(0) for feature in fieldgeom["features"]])

# make a list with unique ids per field to simplify data extraction in the df's
unique_ids_fields = []
for n in range(len(geo)): unique_ids_fields.extend([uuid.uuid4().hex[:30].lower()])

start = '2019-01-01'
end = '2019-12-31'
metrics_order = ['VH','VV', 'fAPAR'] # in case coherence only extract the VV one


bands_ts = get_bands(start,end)


##### POST PROCESSING TIMESERIES USING A UDF
#datacube_metrics = bands_ts.filter_temporal(start,end).polygonal_mean_timeseries(geo).execute_batch(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\TAP_fields_datacube_metrics.json")
#df_metrics = timeseries_json_to_pandas(datacube_metrics)
import json
with open(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\TAP_fields_datacube_metrics.json",'r') as ts_file:
    ts_dict = json.load(ts_file)
    df_metrics = timeseries_json_to_pandas(ts_dict)
    df_metrics.index  = pd.to_datetime(df_metrics.index)

# rename the columns to the name of the metric and the id of the field
def rename_df_columns(df, unique_ids_fields, metrics_order):
    df.columns.set_levels(unique_ids_fields, level=0, inplace=True)
    df.columns.set_levels(metrics_order, level=1, inplace=True)
    df.columns = ['_'.join(col).strip() for col in df_metrics.columns.values]
    return df
df_metrics = rename_df_columns(df_metrics, unique_ids_fields, metrics_order)


#### function to calculate the VHVV ratio for the S1 bands + rescale to values between 0 and 1
VH_VV_range = [-30,-8] # min/max range of the VHVV metric

def VHVV_calc(df, ids_field, VH_VV_range):
    for id in ids_field:
        df['{}_VH_VV'.format(id)] = 10 * np.log(df['{}_VH'.format(id)] / df['{}_VV'.format(id)])
        df['{}_VH_VV'.format(id)]  =  2 * (df['{}_VH_VV'.format(id)] - VH_VV_range[0]) / (VH_VV_range[1] - VH_VV_range[0]) - 1 # rescale
    return df
df_metrics = VHVV_calc(df_metrics,unique_ids_fields, VH_VV_range)

#### rescale the fAPAR
fAPAR_rescale = 0.005

def rescale_metrics(df, fAPAR_rescale, ids_field):
    df[[item + '_fAPAR' for item in unique_ids_fields]] = df.loc[:,df_metrics.columns.isin([item + '_fAPAR' for item in ids_field])]*fAPAR_rescale
    return df
df_metrics = rescale_metrics(df_metrics, fAPAR_rescale, unique_ids_fields)

#### create df structure to allow ingestion in NN model
### for now just extract the ro 110 and 161
t_110 = pd.date_range("2019-01-02","2019-12-31",freq="6D",tz = 'utc').to_pydatetime() # the dates at which this specific orbit occur
t_161 = pd.date_range("2019-01-05","2019-12-31",freq="6D",tz = 'utc').to_pydatetime() # the dates at which this specific orbit occur
ts_orbits= [t_110, t_161]
ro_s = ['ro110','ro161'] # the orbits of consideration
window_values = 5  # define the amount of S1 coverages within the window for extraction
metrics = ['fAPAR','VH_VV']
def prepare_df_NN_model(df, ts_orbits, window_values, ids_field, ro_s, metrics):
    window_width = (window_values - 1) * 6  # days within the window
    df_harvest_model = []
    print('{} FIELDS TO COMPILE IN DATASET'.format(len(ids_field)))
    o = 0
    for ro in ro_s:
        df_orbit = df.reindex(ts_orbits[o])
        moving_window_steps = np.arange(0, df_orbit.shape[0] - window_values - 1) # the amount of windows that can be created on the time period
        for id in ids_field:
            df_id = df_orbit.loc[:,df_orbit.columns.str.contains(id)]
            df_field_input_data_harvest_model = []
            for p in range(len(moving_window_steps)):
                df_id_window = pd.DataFrame(df_id.iloc[p:p + window_values, :])
                middle_date_window =  pd.DataFrame(df_id_window.index[0] + timedelta(window_width/2), index = [id], columns=(['prediction_date_window']))#the center data of the window which can is in fact the harvest prediction date if the model returns 1
                df_id_window = pd.DataFrame(df_id_window.loc[:, df_id_window.columns.isin([id + '_{}'.format(item) for item in metrics])].T.values.flatten()).T # insert the window data as a row in the dataframe
                ###### create list of input metrics of window
                df_id_window = pd.DataFrame(df_id_window)
                df_id_window.index = [id]
                if df_id_window.isnull().values.all() or df_id_window.isnull().values.all(): #if no data in window => continue .any() no use .all() to allow running
                    print('NO DATA FOR {} AND IN ORBIT {}'.format(id,ro))
                    continue
                df_id_window = pd.concat([df_id_window, middle_date_window], axis = 1)
                df_harvest_model.append(df_id_window)
        o += 1
    df_harvest_model = pd.concat(df_harvest_model, axis=0)
    df_harvest_model.index.name = 'ID_field'
    return df_harvest_model

df_input_NN  = prepare_df_NN_model(df_metrics, ts_orbits, window_values, unique_ids_fields, ro_s, metrics)


#### run the NN model
amount_metrics_model = len(metrics)*window_values
thr = 0.8 # threshold for harvest detection
crop_calendar_event = 'Harvest'
def apply_NN_model_crop_calendars(df, amount_metrics_model, thr, crop_calendar_event):
    x_test = df.iloc[0:df.shape[0], 0:amount_metrics_model]
    x_test = x_test.fillna(method='ffill') # fill the empty places
    loaded_model = load_model(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\output\Test10\model_update1.0_iteration0.h5")
    predictions = loaded_model.predict(x_test)
    predictions[predictions >= thr] = 1
    predictions[predictions < thr] = 0
    df['crop_calendar_detection_{}'.format(crop_calendar_event)] = predictions
    return df
df_NN_prediction = apply_NN_model_crop_calendars(df_input_NN, amount_metrics_model, thr, crop_calendar_event)


def create_crop_calendars_fields(df, ids_field):
    df_crop_calendars =[]
    for id in ids_field: ### here can insert a loop for the different crop calendar events for that field
        crop_calendar_date = pd.to_datetime(df[(df['crop_calendar_detection_Harvest'] == 1) & (df.index == id)].prediction_date_window).mean() # take the average of the dates at which a crop event occured according to the model
        df_crop_calendars.append(pd.DataFrame(data = crop_calendar_date, index = [id], columns= ['Harvest_date']))
    df_crop_calendars = pd.concat(df_crop_calendars)
    return df_crop_calendars
df_crop_calendars_result = create_crop_calendars_fields(df_NN_prediction, unique_ids_fields)
