import numpy as np
# import openeo
import pandas as pd
# import scipy.signal
# import shapely
from openeo.rest.conversions import timeseries_json_to_pandas
from openeo_udf.api.udf_data import UdfData
from openeo_udf.api.structured_data import StructuredData
from cropsar.preprocessing.retrieve_timeseries_openeo import run_cropsar_dataframes


from tensorflow.keras.models import load_model
# import geojson
# import uuid
# import json

######## FUNCTIONS ################
def get_cropsar_TS(ts_df, unique_ids_fields, metrics_order, fAPAR_rescale_Openeo, Spark = True):
    index_fAPAR = metrics_order.index('fAPAR')
    df_S2 = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([str(index_fAPAR)])].sort_index().T
    df_S2 *= fAPAR_rescale_Openeo
    index_S1_ascending = metrics_order.index('sigma_ascending_VH')
    df_S1_ascending = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([str(index_S1_ascending), str(index_S1_ascending+1), str(index_S1_ascending +2)])].sort_index().T
    index_S1_descending = metrics_order.index('sigma_descending_VH')
    df_S1_descending = ts_df.loc[:, ts_df.columns.get_level_values(1).isin([str(index_S1_descending), str(index_S1_descending+1), str(index_S1_descending +2)])].sort_index().T
    if Spark:
        cropsar_df, cropsar_df_q10, cropsar_df_q90 = run_cropsar_dataframes(df_S2, df_S1_ascending, df_S1_descending)
        cropsar_df = cropsar_df.rename(columns = dict(zip(list(cropsar_df.columns.values), [item+ '_cropSAR' for item in unique_ids_fields])))
        cropsar_df.index = pd.to_datetime(cropsar_df.index).date
    return cropsar_df
# rename the columns to the name of the metric
# and the id of the field
def rename_df_columns(df, unique_ids_fields, metrics_order):
    df.columns.set_levels(unique_ids_fields, level=0, inplace=True)
    df.columns.set_levels(metrics_order, level=1, inplace=True)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

# function to calculate the VHVV ratio for the S1 bands
# + rescale to values between 0 and 1
def VHVV_calc_rescale(df, ids_field, VH_VV_range):
    for id in ids_field:
        for mode in ['ascending', 'descending']:
            df['{}_VH_VV_{}'.format(id,mode)] = 10 * np.log10(df['{}_sigma_{}_VH'.format(id,mode)] / df['{}_sigma_{}_VV'.format(id,mode)])
            df['{}_VH_VV_{}'.format(id,mode)] = 2 * (df['{}_VH_VV_{}'.format(id,mode)] - VH_VV_range[0]) / (
                    VH_VV_range[1] - VH_VV_range[0]) - 1  # rescale
    return df
# function to rescale the cropsar fAPAR curve
def rescale_cropSAR(df, range, ids_field, metric_suffix):
    df[[item + '_{}'.format(str(metric_suffix)) for item in ids_field]] = 2 * (
            df[[item + '_{}'.format(str(metric_suffix)) for item in ids_field]] - range[0]) / (range[1] -range[0]) - 1
    return df

# function to create df structure that
# allows ingestion in NN model
def prepare_df_NN_model(df, window_values, ids_field, ro_s, metrics_crop_event):
    #local import, file level import has issue in udf inspection
    from datetime import timedelta

    window_width = (window_values - 1) * 6  # days within the window
    df_harvest_model = []
    print('{} FIELDS TO COMPILE IN DATASET'.format(len(ids_field)))
    for id_field in ids_field:
        ts_descending = pd.date_range('{}'.format(list(ro_s['descending']['{}'.format(id_field)].keys())[0]),
                                      '{}-12-31'.format(
                                          list(ro_s['descending']['{}'.format(id_field)].keys())[0].rsplit('-')[0]),
                                      freq="6D", tz='utc').date
        ts_ascending = pd.date_range('{}'.format(list(ro_s['ascending']['{}'.format(id_field)].keys())[0]),
                                     '{}-12-31'.format(
                                         list(ro_s['ascending']['{}'.format(id_field)].keys())[0].rsplit('-')[0]),
                                     freq="6D", tz='utc').date
        ts_orbits = [ts_descending, ts_ascending]
        orbit_pass = [r'descending', r'ascending']
        o = 0
        for ts_orbit in ts_orbits:
            df_orbit = df.reindex(ts_orbit)
            metrics_crop_event_orbit_pass = [item.format(orbit_pass[o]) for item in metrics_crop_event]
            # the amount of windows that can be created in the time period
            moving_window_steps = np.arange(0, df_orbit.shape[
                0] - window_values - 1)
            # TODO DEFINE A PERIOD AROUND THE EVENT OF WHICH WINDOWS WILL BE SAMPLED TO AVOID OFF-SEASON EVENT DETECTION

            ### data juggling so that the data of a window is written in a single row
            # and can be interpreted by the model. The amount of columns per row
            # is determined by the window size and the amount of metrics

            df_id = df_orbit.loc[:, df_orbit.columns.str.contains(id_field)]
            for p in range(len(moving_window_steps)):
                df_id_window = pd.DataFrame(df_id.iloc[p:p + window_values, :])
                # the center date of the window which is in
                # fact the harvest prediction date if the model returns 1
                middle_date_window = pd.DataFrame(df_id_window.index[0] + timedelta(window_width / 2),
                                                  index=[id_field + '_{}'.format(orbit_pass[o])],
                                                  columns=([
                                                      'prediction_date_window']))
                # insert the window data as a row in the dataframe
                df_id_window = pd.DataFrame(df_id_window.loc[:, df_id_window.columns.isin(
                    [id_field + '_{}'.format(item) for item in
                     metrics_crop_event_orbit_pass])].T.values.flatten()).T

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
    # fill the empty places
    x_test = x_test.fillna(method='ffill')
    loaded_model = load_model(NN_model_dir)
    predictions = loaded_model.predict(x_test)
    predictions[predictions >= thr_detection] = 1
    predictions[predictions < thr_detection] = 0
    df['NN_model_detection_{}'.format(crop_calendar_event)] = predictions
    return df

# function to create the crop calendar information for the fields

def create_crop_calendars_fields(df, ids_field, index_window_above_thr):
    # the dataframe the will store per field the predict crop calendar event
    df_crop_calendars = []
    orbit_passes = [r'ascending', r'descending']
    # here can insert a loop for the different
    # crop calendar events for that field
    for id in ids_field:
        # the dataframe that will temporarily store
        # the predicted crop calendar event per orbit pass
        df_crop_calendars_orbit_pass = []
        df_filtered_id = df[df.index.str.contains(id)]
        for orbit_pass in orbit_passes:
            df_filtered_id_pass = df_filtered_id[(df_filtered_id.index.str.contains(orbit_pass)) & (
                        df_filtered_id['NN_model_detection_Harvest'] == 1)]
            if not df_filtered_id_pass.shape[0] < index_window_above_thr + 1:
                df_crop_calendars_orbit_pass.append(pd.DataFrame(data=pd.to_datetime(
                    df_filtered_id_pass.iloc[index_window_above_thr, :]['prediction_date_window']),
                                                                 index=['{}'.format(orbit_pass)], columns=[
                        'prediction_date']))  # select the x-th position for which the threshold was exceeded

        if df_crop_calendars_orbit_pass:
            df_crop_calendars_orbit_pass = pd.concat(df_crop_calendars_orbit_pass)
            crop_calendar_date = pd.to_datetime(df_crop_calendars_orbit_pass.prediction_date).mean()
        else:
            crop_calendar_date = pd.to_datetime(np.nan)

        ## check if no nan date for the event
        if not np.isnan(crop_calendar_date.day):
            crop_calendar_date = crop_calendar_date.strftime('%Y-%m-%d')  # convert to string format
            df_crop_calendars.append(pd.DataFrame(data=crop_calendar_date, index=[id], columns=['Harvest_date']))
        else:
            df_crop_calendars.append(pd.DataFrame(data=np.nan, index=[id], columns=['Harvest_date']))
    df_crop_calendars = pd.concat(df_crop_calendars)
    return df_crop_calendars

def udf_cropcalendars(udf_data:UdfData):
    context_param_var = udf_data.user_context
    print(context_param_var)
    ts_dict = udf_data.get_structured_data_list()[0].data
    if not ts_dict: #workaround of ts_dict is empty
        return
    ts_df = timeseries_json_to_pandas(ts_dict)
    ts_df.index = pd.to_datetime(ts_df.index).date

    # function to calculate the cropsar curve
    ts_df_cropsar = get_cropsar_TS(ts_df, context_param_var.get('unique_ids_fields'), context_param_var.get('metrics_order'), context_param_var.get('fAPAR_rescale_Openeo'))
    # rescale cropsar values
    ts_df_cropsar = rescale_cropSAR(ts_df_cropsar, context_param_var.get('fAPAR_range_normalization'), context_param_var.get('unique_ids_fields'), 'cropSAR')

    # function to rescale the metrics based
    # on the rescaling factor of the metric
    def rescale_metrics(df, rescale_factor, fAPAR_range, unique_ids_fields, metric_suffix):
        df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] = df.loc[:, ts_df.columns.isin(
            [item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields])] * rescale_factor
        df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] = 2 * (
                df[[item + '_{}'.format(str(metric_suffix)) for item in unique_ids_fields]] - fAPAR_range[0]) / (
                                                                                              fAPAR_range[1] -
                                                                                              fAPAR_range[0]) - 1
        return df


    #### USE THE FUNCTIONS TO DETERMINE THE CROP CALENDAR DATES

    ### EVENT 1: HARVEST DETECTION
    NN_model_dir = context_param_var.get('path_harvest_model')
    amount_metrics_model = len(context_param_var.get('metrics_crop_event')) * context_param_var.get('window_values')

    #### PREPARE THE DATAFRAMES (REFORMATTING AND RESCALING) IN THE
    # RIGHT FORMAT TO ALLOW THE USE OF THE TRAINED NN
    ts_df_prepro = rename_df_columns(ts_df, context_param_var.get('unique_ids_fields'), context_param_var.get('metrics_order'))

    ts_df_prepro = VHVV_calc_rescale(ts_df_prepro, context_param_var.get('unique_ids_fields'), context_param_var.get('VH_VV_range_normalization'))

    #### rescale the fAPAR to 0 and 1 and convert
    # it to values between -1 and 1
    ts_df_prepro = rescale_metrics(ts_df_prepro, context_param_var.get('fAPAR_rescale_Openeo'), context_param_var.get('fAPAR_range_normalization'),
                                   context_param_var.get('unique_ids_fields'), 'fAPAR')

    ro_s = {'ascending': context_param_var.get('RO_ascending_selection_per_field'), 'descending': context_param_var.get('RO_descending_selection_per_field')}

    #### now merge the cropsar ts file with the other
    # df containing the S1 metrics
    date_range = pd.date_range(ts_df_cropsar.index[0], ts_df_cropsar.index[-1]).date
    ts_df_prepro = ts_df_prepro.reindex(date_range)  # need to set the index axis on the same frequency
    ts_df_prepro = pd.concat([ts_df_cropsar, ts_df_prepro], axis=1) # the columns of the cropsar df need to be the first ones in the new df to ensure the correct position for applying the NN model

    ### create windows in the time series to extract the metrics
    # and store each window in a seperate row in the dataframe
    ts_df_input_NN = prepare_df_NN_model(ts_df_prepro, context_param_var.get('window_values'), context_param_var.get('unique_ids_fields'), ro_s,
                                         context_param_var.get('metrics_crop_event'))

    ### apply the trained NN model on the window extracts
    df_NN_prediction = apply_NN_model_crop_calendars(ts_df_input_NN, amount_metrics_model, context_param_var.get('thr_detection'),
                                                     context_param_var.get('crop_calendar_event'), NN_model_dir)
    df_crop_calendars_result = create_crop_calendars_fields(df_NN_prediction, context_param_var.get('unique_ids_fields'), context_param_var.get('index_window_above_thr'))
    print(df_crop_calendars_result)
    # return the predicted crop calendar events as a dict  (json format)
    udf_data.set_structured_data_list([StructuredData(description="crop calendar json",data=df_crop_calendars_result.to_dict(),type="dict")])
    return udf_data