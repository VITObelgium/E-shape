import numpy as np
import openeo
import pandas as pd
import scipy.signal
import shapely
from openeo.rest.conversions import timeseries_json_to_pandas
from cropsar.preprocessing.retrieve_timeseries_openeo import run_cropsar_dataframes

from tensorflow.keras.models import load_model
import geojson
import uuid
import json

##### CROP CALENDAR EVENT SPECIFIC PARAMETERS FOR THE EVENT THAT NEEDS TO BE DETERMINED
window_values = 5  # define the amount of S1 coverages within the window for extraction
thr_detection = 0.75  # threshold for crop event detection
index_window_above_thr = 2
crop_calendar_event = 'Harvest'
metrics_crop_event = ['cropSAR', 'VH_VV_{}']  # the metrics used to determine the crop calendar event
VH_VV_range_normalization= [-13, -3.5]
fAPAR_range_normalization= [0,1]
fAPAR_rescale_Openeo= 0.005
coherence_rescale_Openeo= 0.004
max_gap_prediction = 24 # the max amount of days that are allowed between harvest detection predictions
shub  = False


metrics_order =  ['sigma_ascending_VH', 'sigma_ascending_VV','sigma_ascending_angle','sigma_descending_VH', 'sigma_descending_VV','sigma_descending_angle', 'fAPAR']  # The index position of the metrics returned from the OpenEO datacube
  # The index position of the metrics returned from the OpenEO datacube

path_harvest_model=r"C:\Users\bontek\git\e-shape\Pilot1\Tests\Cropcalendars\Model\model_update1.0_iteration24.h5"

######## FUNCTIONS ################

def throw_if_empty(df):
    if(df.empty):
        raise ValueError("Empty input data for cropsar!")

def get_cropsar_TS(ts_df, unique_ids_fields, metrics_order, fAPAR_rescale_Openeo,  shub, Spark = False):
    index_fAPAR = metrics_order.index('fAPAR')
    df_S2 = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([index_fAPAR])].sort_index().T
    throw_if_empty(df_S2)
    if not shub:
        df_S2 *= fAPAR_rescale_Openeo
    index_S1_ascending = metrics_order.index('sigma_ascending_VH')
    df_S1_ascending = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([index_S1_ascending, index_S1_ascending+1, index_S1_ascending +2])].sort_index().T
    throw_if_empty(df_S1_ascending)
    index_S1_descending = metrics_order.index('sigma_descending_VH')
    df_S1_descending = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([index_S1_descending, index_S1_descending+1, index_S1_descending +2])].sort_index().T
    throw_if_empty(df_S1_descending)
    if Spark:
        cropsar_df, cropsar_df_q10, cropsar_df_q90 = run_cropsar_dataframes(df_S2, df_S1_ascending, df_S1_descending)
    else:
        cropsar_df = pd.read_csv(r"S:\eshape\tmp\harvest_detector\Field_BE\Field_BE_TS_20190101_20190731_cropsar.csv")
        cropsar_df = cropsar_df.set_index(cropsar_df.columns[0])
        cropsar_df = cropsar_df.rename(columns = dict(zip(list(cropsar_df.columns.values), [item+ '_cropSAR' for item in unique_ids_fields])))
        cropsar_df.index = pd.to_datetime(cropsar_df.index).date
    return cropsar_df
# rename the columns to the name of the metric and the id of the field
def rename_df_columns(df, unique_ids_fields, metrics_order):
    df.columns.set_levels(unique_ids_fields, level=0, inplace=True)
    df.columns.set_levels(metrics_order, level=1, inplace=True)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

# function to calculate the VHVV ratio for the S1 bands + rescale to values between 0 and 1
def VHVV_calc_rescale(df, ids_field, VH_VV_range):
    for id in ids_field:
        for mode in ['ascending', 'descending']:
            df['{}_VH_VV_{}'.format(id,mode)] = 10 * np.log10(df['{}_sigma_{}_VH'.format(id,mode)] / df['{}_sigma_{}_VV'.format(id,mode)])
            df['{}_VH_VV_{}'.format(id,mode)] = 2 * (df['{}_VH_VV_{}'.format(id,mode)] - VH_VV_range[0]) / (
                    VH_VV_range[1] - VH_VV_range[0]) - 1  # rescale
    return df



# function to create df structure that allows ingestion in NN model
def prepare_df_NN_model(df, window_values, ids_field, ro_s, metrics_crop_event):
    #local import, file level import has issue in udf inspection
    from datetime import timedelta

    window_width = (window_values - 1) * 6  # days within the window
    df_harvest_model = []
    print('{} FIELDS TO COMPILE IN DATASET'.format(len(ids_field)))
    for id_field in ids_field:
        if list(ro_s['descending']['{}'.format(id_field)].keys()):
            ts_descending = pd.date_range('{}'.format(list(ro_s['descending']['{}'.format(id_field)].keys())[0]),
                                            df.index[-1], # '{}-12-31'.format(list(ro_s['descending']['{}'.format(id_field)].keys())[0].rsplit('-')[0])
                                             freq="6D", tz='utc').date
        else:
            ts_descending = None

        if list(ro_s['ascending']['{}'.format(id_field)].keys()):
            ts_ascending = pd.date_range('{}'.format(list(ro_s['ascending']['{}'.format(id_field)].keys())[0]),
                                             df.index[-1], #'{}-12-31'.format(list(ro_s['ascending']['{}'.format(id_field)].keys())[0].rsplit('-')[0])
                                             freq="6D", tz='utc').date
        else:
            ts_ascending = None
        ts_orbits = [ts_descending, ts_ascending]
        orbit_pass = [r'descending',r'ascending']
        o = 0
        for ts_orbit in ts_orbits:
            if ts_orbit is None:
                o += 1
                ## ORBIT IS NOT AVAILABLE FOR HARVEST DETECTION
                continue
            #TODO loop over the orbit passes and change the name of the metrics crop event so that it can be found by the filtering
            df_orbit = df.reindex(ts_orbit)
            metrics_crop_event_orbit_pass = [item.format(orbit_pass[o]) for item in metrics_crop_event]
            moving_window_steps = np.arange(0, df_orbit.shape[
                0] - window_values - 1)  # the amount of windows that can be created in the time period
            # TODO DEFINE A PERIOD AROUND THE EVENT OF WHICH WINDOWS WILL BE SAMPLED TO AVOID OFF-SEASON EVENT DETECTION
            ### data juggling so that the data of a window is written in a single row and can be interpreted by the model. The amount of columns per row is determined by the window size and the amount of metrics

            df_id = df_orbit.loc[:, df_orbit.columns.str.contains(id_field)]
            for p in range(len(moving_window_steps)):
                df_id_window = pd.DataFrame(df_id.iloc[p:p + window_values, :])
                middle_date_window = pd.DataFrame(df_id_window.index[0] + timedelta(window_width / 2), index=[id_field+'_{}'.format(orbit_pass[o])],
                                                  columns=([
                                                      'prediction_date_window']))  # the center date of the window which is in fact the harvest prediction date if the model returns 1
                df_id_window = pd.DataFrame(df_id_window.loc[:, df_id_window.columns.isin(
                    [id_field + '_{}'.format(item) for item in
                     metrics_crop_event_orbit_pass])].T.values.flatten()).T  # insert the window data as a row in the dataframe

                ###### create list of input metrics of window
                df_id_window = pd.DataFrame(df_id_window)
                df_id_window.index = [id_field + '_{}'.format(orbit_pass[o])]
                if df_id_window.isnull().values.all() or df_id_window.isnull().values.all():  # if no data in window => continue .any() no use .all() to allow running
                    print('NO DATA FOR {} AND IN ORBIT {}'.format(id_field, orbit_pass[o]))

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
    df['NN_model_detection_{}'.format(crop_calendar_event)] = predictions
    return df

# function to search which harvest predictions are consecutive and count the amount of consecutive prediction for final harvest detection
def search_conseecutive_harvest_detections(df_filtering, max_gap_prediction, index_window_above_thr):
    ### check for consecutive harvest detections:

    # first calculate the difference in days between the next harvest detection a shift of one position upward is needed for this
    df_filtering.loc[:, 'days_difference_next_detection'] = df_filtering.loc[:,'prediction_date_window'].diff().dt.days.fillna(999,downcast='infer').shift(-1)

    # second calculate the difference in days between the previous harvest detection. This one is needed to locate a harvest prediction at the end of a series of consecutive harvest predictions
    # this end is marked with a large gap in the next good harvest prediction
    df_filtering.loc[:, 'days_difference_previous_detection'] = df_filtering.loc[:,'prediction_date_window'].diff().dt.days.fillna(999, downcast='infer')

    # fill up the last harvest prediction based on second last difference in prediction
    df_filtering['days_difference_next_detection'].iloc[-1] = df_filtering['days_difference_next_detection'].iloc[-2]

    ## now filter both differences based on the max allowed gap between predictions

    df_filtering['days_difference_next_detection_filter'] = df_filtering['days_difference_next_detection'] <= max_gap_prediction
    df_filtering['days_difference_previous_detection_filter'] = df_filtering['days_difference_previous_detection'] <= max_gap_prediction

    # count the amount of consecutive harvest predictions based on the difference with the next harvest prediction
    df_filtering['times_consecutive_threshold_exceeded'] = (df_filtering['days_difference_next_detection'] <= max_gap_prediction).rolling(index_window_above_thr + 1).sum()

    ## search for the situation when a consecutive series of harvest prediction is only "index_window_above_thr -1" but in fact the next prediction is also within the allowed margin.
    # for this the previous difference column is needed
    df_filtering.loc[((df_filtering['times_consecutive_threshold_exceeded'] == index_window_above_thr) & (
            df_filtering['days_difference_previous_detection_filter'] == True) &
            (df_filtering['days_difference_next_detection_filter'] == False)), 'times_consecutive_threshold_exceeded'] = index_window_above_thr + 1

    output = df_filtering['times_consecutive_threshold_exceeded'].values
    return output


# function to create the crop calendar information for the fields
def create_crop_calendars_fields(df, ids_field, index_window_above_thr, max_gap_prediction):
    df_crop_calendars = [] # the dataframe the will store per field the predict crop calendar event
    orbit_passes = [r'ascending', r'descending']
    for id in ids_field:  ### here can insert a loop for the different crop calendar events for that field
        df_crop_calendars_orbit_pass = []  # the dataframe that will temporarily store the predicted crop calendar event per orbit pass
        #TODO should be updated because now the mean date for both ascending and descending orbits are merged
        df_filtered_id = df[df.index.str.contains(id)]
        for orbit_pass in orbit_passes:
            df_filtered_id_pass = df_filtered_id[(df_filtered_id.index.str.contains(orbit_pass)) & (df_filtered_id['NN_model_detection_Harvest'] == 1)]

            if not df_filtered_id_pass.shape[0] <index_window_above_thr + 1:
                ## filter on the suitable harvest detections (based on the amount of times harvest is predicted within the max_gap_prediction threshold
                count_harvest_consecutive_exceeded = search_conseecutive_harvest_detections(df_filtered_id_pass,
                                                                                            max_gap_prediction,
                                                                                            index_window_above_thr)

                df_filtered_id_pass['count_harvest_consecutive_exceeded'] = count_harvest_consecutive_exceeded

                # select the harvest date the the first time match the requirements
                df_crop_calendars_orbit_pass.append(pd.DataFrame(data=pd.to_datetime
                (df_filtered_id_pass.loc[df_filtered_id_pass['count_harvest_consecutive_exceeded'] == index_window_above_thr+1]['prediction_date_window'].iloc[0]),
                                                          index = ['{}'.format(orbit_pass)], columns= ['prediction_date']))

                # df_crop_calendars_orbit_pass.append(pd.DataFrame(data = pd.to_datetime(df_filtered_id_pass.iloc[index_window_above_thr,:]['prediction_date_window']),
                #                                           index = ['{}'.format(orbit_pass)], columns= ['prediction_date'])) # select the x-th position for which the threshold was exceeded

        if df_crop_calendars_orbit_pass:
            df_crop_calendars_orbit_pass = pd.concat(df_crop_calendars_orbit_pass)
            crop_calendar_date = pd.to_datetime(df_crop_calendars_orbit_pass.prediction_date).mean()
        else:
            crop_calendar_date = pd.to_datetime(np.nan)

        if not np.isnan(crop_calendar_date.day):  ## check if no nan date for the event
            crop_calendar_date = crop_calendar_date.strftime('%Y-%m-%d')  # convert to string format
            df_crop_calendars.append(pd.DataFrame(data=crop_calendar_date, index=[id], columns=['Harvest_date']))
        else:
            df_crop_calendars.append(pd.DataFrame(data = np.nan, index = [id], columns= ['Harvest_date']))
    df_crop_calendars = pd.concat(df_crop_calendars)
    return df_crop_calendars
def rescale_cropSAR(df, range, ids_field, metric_suffix):
    df[[item + '_{}'.format(str(metric_suffix)) for item in ids_field]] = 2 * (
            df[[item + '_{}'.format(str(metric_suffix)) for item in ids_field]] - range[0]) / (
                                                                                          range[1] -
                                                                                          range[0]) - 1
    return df

def udf_cropcalendars_local(ts_dict):
    ts_df = timeseries_json_to_pandas(ts_dict)
    ts_df.index = pd.to_datetime(ts_df.index).date

    some_item_for_date = next(iter(ts_dict.values()))
    unique_ids_fields = ['Field_' + str(item) for item in np.arange(len(some_item_for_date))]

    #unique_ids_fields = [str(uuid.uuid1()) for i in range(number_of_fields)]

    # function to calculate the cropsar curve
    ts_df_cropsar = get_cropsar_TS(ts_df, unique_ids_fields, metrics_order, fAPAR_rescale_Openeo, shub)
    # rescale cropsar values
    ts_df_cropsar = rescale_cropSAR(ts_df_cropsar, fAPAR_range_normalization, unique_ids_fields, 'cropSAR')

    # function to rescale the metrics based on the rescaling factor of the metric
    def rescale_metrics(df, rescale_factor, fAPAR_range, ids_field, metric_suffix, shub):
        if not shub:
            df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] = df.loc[:, ts_df.columns.isin(
                [item + '_{}'.format(str(metric_suffix)) for item in ids_field])] * rescale_factor
        df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] = 2 * (
                df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] - fAPAR_range[0]) / (
                                                                                              fAPAR_range[1] -
                                                                                              fAPAR_range[0]) - 1
        return df
    #### USE THE FUNCTIONS TO DETERMINE THE CROP CALENDAR DATES

    ### EVENT 1: HARVEST DETECTION
    NN_model_dir = path_harvest_model
    amount_metrics_model = len(metrics_crop_event) * window_values

    ### DEFINE THE SUITABLE ORBITS (ASCENDING/DESCENDING) FOR HARVEST DETECTION
    def find_optimal_RO_SHUB(ts_df, orbit_passes, s, shub):
        # TODO USE THE SAME FUNCTION FOR FINDING ORBITS WITH TERRASCOPE DATA
        dict_metadata_ascending_RO_selection = {}
        dict_metadata_descending_RO_selection = {}
        for orbit_pass in orbit_passes:
            angle_fields = ts_df.loc[:,(slice(None), metrics_order.index('sigma_{}_angle'.format(orbit_pass.lower())))]
            angle_fields.columns = angle_fields.columns.get_level_values(0)
            angle_pass_TS = angle_fields.iloc[:, s]
            angle_pass_TS.index = pd.to_datetime(angle_pass_TS.index)
            angle_pass_TS = angle_pass_TS.tz_localize(None)
            # make dataframe from incidence angle pd.series
            df_angle_pass = pd.DataFrame(data=angle_pass_TS.values, columns=(['RO']),
                                         index=angle_pass_TS.index)
            if not shub:
                df_angle_pass = df_angle_pass * 0.0005 +29
            ## round the angles to find which belongs to the same RO
            df_angle_pass['RO_rounded'.format(orbit_pass)] = df_angle_pass['RO'].round(decimals=1)
            unique_angles = list(df_angle_pass.dropna()['RO_rounded'].unique())
            # if not orbit pass available just skip it
            if not unique_angles:
                if orbit_pass == 'ASCENDING':
                    dict_metadata_ascending_RO_selection = {}
                else:
                    dict_metadata_descending_RO_selection = {}
                continue

            dict_dates_angle = dict()
            for angle in unique_angles:
                difference = df_angle_pass['RO'] - angle
                dict_dates_angle.update({angle: list(
                    df_angle_pass.loc[difference[((difference < 0.5) & (difference > -0.5))].index][
                        'RO_rounded'].index.values)})

            RO_orbit_counter = {key: len(value) for key, value in dict_dates_angle.items()}
            RO_shallowest_angle = max(dict_dates_angle.keys())
            # see if the orbit with shallowest angle has not a lot fewer coverages compared to the orbit with the maximum coverages. In case this orbit has more than 80% less
            # coverage another orbit is selected
            if RO_orbit_counter.get(RO_shallowest_angle) < int(max(list(RO_orbit_counter.values())) * 0.80):
                RO_orbit_selection = max(RO_orbit_counter, key=lambda x: RO_orbit_counter[x])
            else:
                RO_orbit_selection = RO_shallowest_angle
            list_orbit_passes = sorted(dict_dates_angle.get(RO_orbit_selection))
            if orbit_pass == 'ASCENDING':
                dict_metadata_ascending_RO_selection = {
                    pd.to_datetime(list_orbit_passes[0]).strftime('%Y-%m-%d'): RO_orbit_selection}
            else:
                dict_metadata_descending_RO_selection = {
                    pd.to_datetime(list_orbit_passes[0]).strftime('%Y-%m-%d'): RO_orbit_selection}

        return dict_metadata_ascending_RO_selection, dict_metadata_descending_RO_selection

    RO_ascending_selection_per_field = dict()
    RO_descending_selection_per_field = dict()
    for s in range(len(unique_ids_fields)):
        RO_ascending_selection, RO_descending_selection = find_optimal_RO_SHUB(ts_df, ['ASCENDING','DESCENDING'], s, shub)
        RO_ascending_selection_per_field.update({'Field_' + str(s): RO_ascending_selection})
        RO_descending_selection_per_field.update({'Field_' + str(s): RO_descending_selection})




    #### PREPARE THE DATAFRAMES (REFORMATTING AND RESCALING) IN THE RIGHT FORMAT TO ALLOW THE USE OF THE TRAINED NN
    ts_df_prepro = rename_df_columns(ts_df, unique_ids_fields, metrics_order)

    ts_df_prepro = VHVV_calc_rescale(ts_df_prepro, unique_ids_fields, VH_VV_range_normalization)

    #### rescale the fAPAR to 0 and 1 and convert it to values between -1 and 1
    ts_df_prepro = rescale_metrics(ts_df_prepro, fAPAR_rescale_Openeo, fAPAR_range_normalization,
                                   unique_ids_fields, 'fAPAR', shub)

    ### for now just extract the ro 110 and 161 for S1_VV and S1_VH
    # TODO USE HERE THE INFORMATION OF MOST FREQUENT RO PER FIELD
    #t_110 = pd.date_range("2019-01-02", "2019-12-31", freq="6D",
                          #tz='utc').to_pydatetime()  # the dates at which this specific orbit occur in BE
    #t_161 = pd.date_range("2019-01-05", "2019-12-31", freq="6D",
                          #tz='utc').to_pydatetime()  # the dates at which this specific orbit occur in BE


    #ro_s = ['ro110', 'ro161']  # the orbits of consideration
    ro_s = {'ascending':RO_ascending_selection_per_field, 'descending': RO_descending_selection_per_field}

    #### now merge the cropsar ts file with the other df containing the S1 metrics
    date_range = pd.date_range(ts_df_cropsar.index[0], ts_df_cropsar.index[-1]).date
    ts_df_prepro = ts_df_prepro.reindex(date_range) # need to set the index axis on the same frequency
    ts_df_prepro = pd.concat([ts_df_cropsar, ts_df_prepro], axis = 1)

    ### create windows in the time series to extract the metrics and store each window in a seperate row in the dataframe
    ts_df_input_NN = prepare_df_NN_model(ts_df_prepro, window_values, unique_ids_fields, ro_s,
                                         metrics_crop_event)

    ### apply the trained NN model on the window extracts
    df_NN_prediction = apply_NN_model_crop_calendars(ts_df_input_NN, amount_metrics_model, thr_detection,
                                                     crop_calendar_event, NN_model_dir)

    df_crop_calendars_result = create_crop_calendars_fields(df_NN_prediction, unique_ids_fields, index_window_above_thr, max_gap_prediction)

    udf_data.set_structured_data_list([StructuredData(description="crop calendar json",data=df_crop_calendars_result.to_dict(),type="dict")])
    return udf_data