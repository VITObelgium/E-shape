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
import geojson
from pathlib import Path
import uuid

# General approach:
#
# first merge all required inputs into a single multiband raster datacube
# compute timeseries for one or more fields, for all bands (one step)
# do postprocessing of the timeseries:
#   compute cropsar based on cleaned fapar + sigma0
#   combine cropsar + coherence to determine cropcalendar
#   return cropcalendar output in your own json format

class Cropcalendars():
    def __init__(self, fAPAR_rescale_Openeo, coherence_rescale_Openeo, path_harvest_model,VH_VV_range_normalization, fAPAR_range_normalization, metrics_order):
        # crop calendar independant variables
        self.fAPAR_rescale_Openeo = fAPAR_rescale_Openeo
        self.coherence_rescale_Openeo = coherence_rescale_Openeo
        self.path_harvest_model = path_harvest_model
        self.VH_VV_range_normalization = VH_VV_range_normalization
        self.fAPAR_range_normalization  = fAPAR_range_normalization
        self.metrics_order = metrics_order

    def generate_cropcalendars(self, start, end, gjson_path, window_values, thr_detection, crop_calendar_event, metrics_crop_event):
            ##### FUNCTION TO BUILD A DATACUBE IN OPENEO

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


            #### LOAD THE FIELDS FOR WHICH THE TIMESERIES SHOULD BE EXTRACTED FOR THE CROP CALENDARS

            ### ingest the field(s) for which the crop calendars should be defined
            #TODO: The properties of the fields should be ingested from the main script => use Geojson as input
            ##### GEOJSON AS INPUT FORMAT FOR EXTRACTING THE DATA
            with open(gjson_path) as f: gj = geojson.load(f)

            ## give every parcel an unique id and convert it to a geometry collection
            unique_ids_fields = []
            for i in range(len(gj)):
                gj.features[i].properties['id'] = str(uuid.uuid1())
                unique_ids_fields.extend([gj.features[i].properties['id']])


            geo=shapely.geometry.GeometryCollection([shapely.geometry.shape(feature["geometry"]).buffer(0) for feature in gj["features"]])

            # # make a list with unique ids per field to simplify data extraction in the df's and to link the crop calendar result with the field
            #for n in range(len(geo)): unique_ids_fields.extend([uuid.uuid4().hex[:30].lower()])

            # get the datacube containing the time series data
            bands_ts = get_bands(start,end)


            ##### POST PROCESSING TIMESERIES USING A UDF
            timeseries = bands_ts.filter_temporal(start,end).polygonal_mean_timeseries(geo)

            def udf_cropcalendars(ts_df, unique_ids_fields):
                #ts_dict = udf_data.get_structured_data_list()[0]
                #ts_df = timeseries_json_to_pandas(ts_dict)
                #ts_df.index = pd.to_datetime(ts_df.index)


                ######## FUNCTIONS ################
                # rename the columns to the name of the metric and the id of the field
                def rename_df_columns(df, unique_ids_fields, metrics_order):
                    df.columns.set_levels(unique_ids_fields, level=0, inplace=True)
                    df.columns.set_levels(metrics_order, level=1, inplace=True)
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    return df

                # function to calculate the VHVV ratio for the S1 bands + rescale to values between 0 and 1
                def VHVV_calc_rescale(df, ids_field, VH_VV_range):
                    for id in ids_field:
                        df['{}_VH_VV'.format(id)] = 10 * np.log10(df['{}_sigma_VH'.format(id)] / df['{}_sigma_VV'.format(id)])
                        df['{}_VH_VV'.format(id)] = 2 * (df['{}_VH_VV'.format(id)] - VH_VV_range[0]) / (
                                VH_VV_range[1] - VH_VV_range[0]) - 1  # rescale
                    return df

                # function to rescale the metrics based on the rescaling factor of the metric
                def rescale_metrics(df, rescale_factor, fAPAR_range, ids_field, metric_suffix):
                    df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] = df.loc[:, df_metrics.columns.isin(
                        [item + '_{}'.format(str(metric_suffix)) for item in ids_field])] * rescale_factor
                    df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] = 2 * (df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] - fAPAR_range[0]) / (fAPAR_range[1] - fAPAR_range[0]) - 1
                    return df

                # function to create df structure that allows ingestion in NN model
                def prepare_df_NN_model(df, ts_orbits, window_values, ids_field, ro_s, metrics_crop_event):
                    window_width = (window_values - 1) * 6  # days within the window
                    df_harvest_model = []
                    print('{} FIELDS TO COMPILE IN DATASET'.format(len(ids_field)))
                    o = 0
                    for ro in ro_s:
                        df_orbit = df.reindex(ts_orbits[o])
                        moving_window_steps = np.arange(0, df_orbit.shape[
                            0] - window_values - 1)  # the amount of windows that can be created in the time period
                        #TODO DEFINE A PERIOD AROUND THE EVENT OF WHICH WINDOWS WILL BE SAMPLED TO AVOID OFF-SEASON EVENT DETECTION
                        ### data juggling so that the data of a window is written in a single row and can be interpreted by the model. The amount of columns per row is determined by the window size and the amount of metrics
                        for id in ids_field:
                            df_id = df_orbit.loc[:, df_orbit.columns.str.contains(id)]
                            for p in range(len(moving_window_steps)):
                                df_id_window = pd.DataFrame(df_id.iloc[p:p + window_values, :])
                                middle_date_window = pd.DataFrame(df_id_window.index[0] + timedelta(window_width / 2), index=[id],
                                                                  columns=([
                                                                      'prediction_date_window']))  # the center date of the window which is in fact the harvest prediction date if the model returns 1
                                df_id_window = pd.DataFrame(df_id_window.loc[:, df_id_window.columns.isin(
                                    [id + '_{}'.format(item) for item in
                                     metrics_crop_event])].T.values.flatten()).T  # insert the window data as a row in the dataframe

                                ###### create list of input metrics of window
                                df_id_window = pd.DataFrame(df_id_window)
                                df_id_window.index = [id]
                                if df_id_window.isnull().values.all() or df_id_window.isnull().values.all():  # if no data in window => continue .any() no use .all() to allow running
                                    print('NO DATA FOR {} AND IN ORBIT {}'.format(id, ro))
                                    continue
                                df_id_window = pd.concat([df_id_window, middle_date_window], axis=1)
                                df_harvest_model.append(df_id_window)
                        o += 1
                    df_harvest_model = pd.concat(df_harvest_model, axis=0)
                    df_harvest_model.index.name = 'ID_field'
                    return df_harvest_model

                # function to run the NN model
                def apply_NN_model_crop_calendars(df, amount_metrics_model, thr_detection, crop_calendar_event, NN_model_dir):
                    x_test = df.iloc[0:df.shape[0], 0:amount_metrics_model]
                    x_test = x_test.fillna(method='ffill')  # fill the empty places
                    loaded_model = load_model(NN_model_dir)
                    predictions = loaded_model.predict(x_test)
                    predictions[predictions >= thr_detection] = 1
                    predictions[predictions < thr_detection] = 0
                    df['crop_calendar_detection_{}'.format(crop_calendar_event)] = predictions
                    return df

                 # function to create the crop calendar information for the fields
                def create_crop_calendars_fields(df, ids_field):
                    df_crop_calendars = []
                    for id in ids_field:  ### here can insert a loop for the different crop calendar events for that field
                        crop_calendar_date = pd.to_datetime(df[(df['crop_calendar_detection_Harvest'] == 1) & (
                                df.index == id)].prediction_date_window).mean()  # take the average of the dates at which a crop event occured according to the model #TODO adapt this method based on analysis results
                        if not np.isnan(crop_calendar_date.day): ## check if no nan date for the event
                            crop_calendar_date = crop_calendar_date.strftime('%Y-%m-%d') # convert to string format
                        df_crop_calendars.append(pd.DataFrame(data=crop_calendar_date, index=[id], columns=['Harvest_date']))
                    df_crop_calendars = pd.concat(df_crop_calendars)
                    return df_crop_calendars

               #### USE THE FUNCTIONS TO DETERMINE THE CROP CALENDAR DATES

                ### EVENT 1: HARVEST DETECTION
                NN_model_dir = self.path_harvest_model
                amount_metrics_model = len(metrics_crop_event) * window_values

                #### ADRESS THE FUNCTION TO DETERMINE THE HARVEST DATE FOR THE FIELDS
                ts_df_prepro = rename_df_columns(ts_df, unique_ids_fields, self.metrics_order)

                ts_df_prepro = VHVV_calc_rescale(ts_df_prepro, unique_ids_fields, self.VH_VV_range_normalization)

                #### rescale the fAPAR to 0 and 1 and convert it to values between -1 and 1
                ts_df_prepro = rescale_metrics(ts_df_prepro, self.fAPAR_rescale_Openeo, self.fAPAR_range_normalization, unique_ids_fields, 'fAPAR')


                ### for now just extract the ro 110 and 161
                #TODO In the final script the RO should be dynamically chosen based on the overlap of descending and ascending orbits of the parcel
                t_110 = pd.date_range("2019-01-02", "2019-12-31", freq="6D",
                                      tz='utc').to_pydatetime()  # the dates at which this specific orbit occur in BE
                t_161 = pd.date_range("2019-01-05", "2019-12-31", freq="6D",
                                      tz='utc').to_pydatetime()  # the dates at which this specific orbit occur in BE
                ts_orbits = [t_110, t_161]
                ro_s = ['ro110', 'ro161']  # the orbits of consideration

                ts_df_input_NN = prepare_df_NN_model(ts_df_prepro, ts_orbits, window_values, unique_ids_fields, ro_s, metrics_crop_event)

                df_NN_prediction = apply_NN_model_crop_calendars(ts_df_input_NN, amount_metrics_model, thr_detection, crop_calendar_event,NN_model_dir)

                df_crop_calendars_result = create_crop_calendars_fields(df_NN_prediction, unique_ids_fields)

                #udf_data.set_structured_data_list([df_crop_event.to_dict()])
                return df_crop_calendars_result



            # cropcalendar_result = timeseries.process("run_udf",
            #                                          data = timeseries._pg, udf = udf, runtime = 'Python').execute_batch(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\crop_calendar_field_test.json")

            #datacube_metrics = bands_ts.filter_temporal(start,end).polygonal_mean_timeseries(geo).execute_batch(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\TAP_fields_datacube_metrics_test.json")
            #df_metrics = timeseries_json_to_pandas(datacube_metrics)
            import json
            with open(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\TAP_fields_datacube_metrics_test.json",'r') as ts_file:
                ts_dict = json.load(ts_file)
                df_metrics = timeseries_json_to_pandas(ts_dict)
                df_metrics.index  = pd.to_datetime(df_metrics.index)

            crop_calendars = udf_cropcalendars(df_metrics, unique_ids_fields)

            #### FINALLY ASSIGN THE CROP CALENDAR EVENTS AS PROPERTIES TO THE GEOJSON FILE WITH THE FIELDS
            for s in range(len(gj)):
                for c in range(crop_calendars.shape[1]):  # the amount of crop calendar events which were determined
                    gj.features[s].properties[crop_calendars.columns[c]] = \
                    crop_calendars.loc[crop_calendars.index == unique_ids_fields[s]][crop_calendars.columns[c]].values[0]  # the date of the event


            return gj








