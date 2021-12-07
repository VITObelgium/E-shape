import pandas as pd
import os
import glob
import numpy as np
from scipy.signal import savgol_filter
import datetime



def filter_orbits(files, ID, orbit_pass = 'ASCENDING'):
    """
    Function that will be used to only load the most shallowest incidence angle for each orbit pass
    :param files: the files containing the TS of the dataset of consideration
    :param ID: the ID of the field needed to filter the dataframe
    :param orbit_pass: indicate which orbit is needed to be considered for filtering
    :return: a dataframe containing the TS for the shallowest incidence angle of that specific orbit
    """

    incidence_angles = [float(os.path.split(item)[-1].split('_')[-2]) for item in files if orbit_pass in item]

    incidence_angles.sort(reverse = True)
    dict_count_obs_angle = dict()
    dict_files_ID_angle = dict()
    for angle in incidence_angles:
        file_select = [item for item in files if os.path.split(item)[-1].split('_')[-2] == str(angle) ]
        df_test = pd.read_csv(file_select[0], header = [0,1], index_col =  0)
        try:
            df_test = df_test[ID]
        except:
            continue
        df_test = df_test.dropna(how = 'all', axis = 1) # drop empty columns
        if df_test.shape[-1]<=1: #sometimes only values for incidence angle are available
            continue

        if df_test.dropna().empty or (df_test.isna().sum()[0]/ df_test.shape[0])*100 >30: #if more than 30% nan check next incidence angle
            df_final = None
            continue
        else:
            dict_count_obs_angle.update({angle: df_test.dropna().shape[0]})
            dict_files_ID_angle.update({angle: df_test})
            continue
            #df_final = df_test
            #break
    #check if shallowest incidenge angle contains enough valid data
    incidence_angles_data = sorted(list(dict_count_obs_angle.keys()), reverse= True)
    if len(incidence_angles_data) >1:
        if dict_count_obs_angle.get(incidence_angles_data[0]) >= max(dict_count_obs_angle.values())*0.9:
            df_final = dict_files_ID_angle.get(incidence_angles_data[0])
        else:
            if dict_count_obs_angle.get(incidence_angles_data[1]) >= max(dict_count_obs_angle.values())*0.8:
                df_final = dict_files_ID_angle.get(incidence_angles_data[1])
            else:
                angle_max_obs = max(dict_count_obs_angle, key = dict_count_obs_angle.get)
                df_final = dict_files_ID_angle.get(angle_max_obs)
    elif len(incidence_angles_data) == 1:
        df_final = dict_files_ID_angle.get(incidence_angles_data[0])
    else:
        return None



    return df_final


def calc_VH_VV(df_input, orbit_pass = 'ASCENDING'):
    """
    Function used to calculate the VH/VV ratio on S1 TS
    :param df_input: dataframe with TS of VH and VV
    :param orbit_pass: identify for which orbit pass this needs to be done
    :return: the input dataframe with as additional column the VH/VV ratio
    """
    df_input['VH/VV'] = 10 * np.log((df_input['sigma_{}_VH'.format(orbit_pass.lower())] )
                                    / ((df_input['sigma_{}_VV'.format(orbit_pass.lower())] )))
    return df_input

def to_dB(df_input, metric):
    """
    Function to convert a backscatter coefficient to decibels
    :param df_input: dataframe with the TS of a metric (see parameter metric)
    :param metric: the name of the metric that should be converted to decibels
    :return: the same dataframe, but with the metric now converted to decibels
    """
    df_input[metric] = 10 * np.log(df_input[metric])
    return df_input

def plot_generator(df, ax1, row, col, color, metric, row_only = False, S1_VHVV = False, S2_bands = False, cropsar = False, S1_indv_polar = False,S1_coherence = False,
                   in_situ = None,orbit_pass = 'ASCENDING', S1_interp  = False):

    if row_only:

        if cropsar:
            ax1[row].plot(df['q50'], fillstyle='full', color=color, label='CropSAR_q50')
            ax1[row].plot(df['q10'], linestyle = '-.', fillstyle='full', color=color, label='CropSAR_q10')
            ax1[row].plot(df['q90'], linestyle = '--', fillstyle='full', color=color, label='CropSAR_q90')
            ax1[row].set_ylabel('fAPAR')


        if S1_VHVV:
            df['VH/VV' + '_filled']  = df['VH/VV'].asfreq('D').interpolate(method = 'time')
            ax1[row].scatter(x = df.index, y  =df['VH/VV'], marker = 'o', color=color, label='VH/VV_{}'.format(orbit_pass) )
            ax1[row].plot( df['VH/VV_filled'], color=color, label='VH/VV_{}_filled'.format(orbit_pass))
            ax1[row].set_ylabel('VH/VV (dB)')

        if S1_indv_polar and not S1_interp:
            df[metric + '_filled']  = df[metric].asfreq('D').interpolate(method = 'time')
            ax1[row].scatter(x = df.index, y  =df[metric], marker = 'o', color=color)
            ax1[row].plot(df[metric + '_filled'], color=color, label='{}_{}_filled'.format(metric, orbit_pass))
            ax1[row].set_ylabel('Backscattering (dB)')
        if S1_indv_polar and S1_interp:
            ax1[row].plot(df[metric], color=color, label='{}_{}_filled'.format(metric, orbit_pass))
            ax1[row].set_ylabel('Backscattering (dB)')

        if S1_coherence and not S1_interp:
            df[metric + '_filled']  = df[metric].asfreq('D').interpolate(method = 'time')
            ax1[row].scatter(x = df.index, y  =df[metric], marker = 'o', color=color)
            ax1[row].plot(df[metric + '_filled'], color=color, label='{}_{}_filled'.format(metric, orbit_pass))
            ax1[row].set_ylabel('Coherence')
        if S1_coherence and S1_interp:
            ax1[row].plot(df[metric], color=color, label='{}_{}_filled'.format(metric, orbit_pass))
            ax1[row].set_ylabel('Coherence')




        if S2_bands:
            df[metric + '_filled']  = df[metric].asfreq('D').interpolate(method = 'time')
            ax1[row].plot(df[metric + '_filled'], fillstyle = 'full', color = color)
            ax1[row].scatter(x = df.index, y = df[metric], marker = 'o', color = color, label = metric)

            ax1[row].set_ylabel('Reflectance')

        if in_situ is not None:
            for date in in_situ:
                if date != None:
                    ax1[row].axvline(date, linestyle = '--', color = 'black')



        ax1[row].legend(loc='upper right')
        ax1[row].set_xlim([df.index.values[0], df.index.values[-1]])

    else:
        print('DOES NOT APPLY FOR THE MOMENT')
        # ax1[row, col].plot(df[metric], marker='o', fillstyle='full', color=color, label=metric)
        # ax1[row, col].plot(df[metric + '_filled'], linestyle='--', fillstyle='full', color=color,
        #                    )#label=metric + '_filled'
        # ax1[row, col].legend(loc='upper left')
        # if not S2_bands and not coherence:
        #     ax1[row, col].set_ylabel(metric)
        # elif S2_bands:
        #     ax1[row, col].set_ylabel('Reflectance')
        # elif coherence:
        #     ax1[row, col].set_ylabel('Coherence')


def filter_lst_RO(lst_files):
    """
    Function used to check which S1 files contains incidence information
    :param lst_files: list of S1 files for the dataset of consideration
    :return: list containing only S1 TS for specific RO
    """
    files_filtered_RO = []
    for file in lst_files:
        try:
            if type(float(os.path.split(file)[-1].split('_')[-2])) == float:
                files_filtered_RO.append(file)
        except:
            continue
    return files_filtered_RO

def filter_rain_S1(df, meteo_files, ID, rain_thr = 4):
    """
    Function used to mask the S1 observation at days with too much rainfall

    :param df: input dataframe with S1 TS
    :param meteo_files: location of meteo file for that dataset
    :param ID: the ID of the location for which the meteo data should be filtered
    :param rain_thr: the rain threshold used to mask S1 observations for rain
    :return: a dataframe masked for intense rainfall events
    """
    df_meteo = pd.read_csv(meteo_files[0], index_col = 0)
    df_meteo = df_meteo[[ID]]
    df_meteo.index = pd.to_datetime(df_meteo.index)
    df_meteo.index = [item.replace(tzinfo = None) for item in df_meteo.index]
    df.index = pd.to_datetime(df.index)

    df_meteo_reindex = df_meteo.reindex(df.index)
    dates_rain = df_meteo_reindex[df_meteo_reindex>rain_thr].dropna()
    if dates_rain.empty:
        return df
    df_filtered = df.drop(dates_rain.index.values)
    return df_filtered


def preprocess_S1_data(indir_satellite_data, ID, key_file, coherence = False, rain_filter = False):
    """
    Function used to preprocess the S1 TS in order that they can be used to apply a crop calendar detection on it
    :param indir_satellite_data:  the basefolder needed to locate all the downloaded TS files
    :param ID: the ID of the field needed to locate the appropriate file
    :param key_file: the key name of the dataset for which the extractions were done, needed to locate the folder location
    :param coherence: indicate if coherence needs to be preprocessed else S1 will be preprocessed
    :param rain_filter: indicate of the S1 data should be filtered for rainfall events
    :return: a dataframe with preprocessed TS of S1 or coherence
    """

    if not coherence:
        S1_files_dataset = glob.glob(os.path.join(indir_satellite_data, 'S1_data', key_file, '*.csv'))
    else:
        S1_files_dataset = glob.glob(os.path.join(indir_satellite_data, 'S1_coh', key_file, '*.csv'))

    #check that only RO files are loaded:
    S1_files_dataset = filter_lst_RO(S1_files_dataset)
    year_interest = key_file.split('_')[-1]
    date_range = pd.date_range('{}-01-01'.format(year_interest), '{}-12-31'.format(str(int(year_interest))), freq= '1D')

    df_ascending_ID = filter_orbits(S1_files_dataset, ID,  orbit_pass= 'ASCENDING')
    df_descending_ID = filter_orbits(S1_files_dataset, ID, orbit_pass= 'DESCENDING')
    if not coherence:
        if not df_ascending_ID is None:
            df_ascending_ID = calc_VH_VV(df_ascending_ID, orbit_pass= 'ASCENDING')
            df_ascending_ID = df_ascending_ID.rename(columns = {'sigma_ascending_VH': 'VH'})
            df_ascending_ID.index = pd.to_datetime(df_ascending_ID.index, format= '%Y-%m-%d')
            df_ascending_ID = df_ascending_ID.reindex(date_range)
            if not coherence:
                df_ascending_ID = to_dB(df_ascending_ID, 'VH')


        if not df_descending_ID is None:
            df_descending_ID = calc_VH_VV(df_descending_ID, orbit_pass= 'DESCENDING')
            df_descending_ID = df_descending_ID.rename(columns = {'sigma_descending_VH': 'VH'})
            df_descending_ID.index = pd.to_datetime(df_descending_ID.index, format= '%Y-%m-%d')
            df_descending_ID = df_descending_ID.reindex(date_range)
            if not coherence:
                df_descending_ID = to_dB(df_descending_ID, 'VH')



    if rain_filter:
        meteo_files = glob.glob(os.path.join(indir_satellite_data, 'Meteo', key_file,'*.csv'))
        if not df_descending_ID is None and not df_ascending_ID is None:
            df_ascending_ID = filter_rain_S1(df_ascending_ID, meteo_files, ID)
            df_descending_ID = filter_rain_S1(df_descending_ID, meteo_files, ID)

    return df_ascending_ID, df_descending_ID


def preprocess_S2_data(indir_satellite_data, ID, key_file):
    """
    Function used to preprocess the S2 TS so that they can be used to apply a crop calendar detection on it
    :param indir_satellite_data: the basefolder needed to locate all the downloaded TS files
    :param ID: the ID of the field needed to locate the appropriate file
    :param key_file: the key name of the dataset for which the extractions were done, needed to locate the folder location
    :return: a dataframe with preprocessed TS of S2
    """

    S2_files_dataset = glob.glob(os.path.join(indir_satellite_data, 'S2_data', key_file, '*.csv'))
    year_interest = key_file.split('_')[-1]
    date_range = pd.date_range('{}-01-01'.format(year_interest), '{}-12-31'.format(str(int(year_interest))), freq= '1D')


    df_S2 = pd.read_csv(S2_files_dataset[0], header = [0,1], index_col = 0)
    df_s2_ID = df_S2[ID]
    df_s2_ID = df_s2_ID.drop(columns = ['SCL'])
    df_s2_ID = df_s2_ID * 0.0001
    df_s2_ID.index = pd.to_datetime(df_s2_ID.index, format= '%Y-%m-%d')
    df_s2_ID = df_s2_ID.reindex(date_range)

    return df_s2_ID

def prepare_cropsar_data(indir_satellite_data, ID, key_file):
    """
    Function used to preprocess the CROPSAR TS so that they can be used to apply a crop calendar detection on it

    :param indir_satellite_data: the basefolder needed to locate all the downloaded TS files
    :param ID: the ID of the field needed to locate the appropriate file
    :param key_file: the key name of the dataset for which the extractions were done, needed to locate the folder location
    :return: the dataframe containing the cropsar curve for that ID
    """
    Cropsar_files_dataset = glob.glob(os.path.join(indir_satellite_data, 'CropSAR',
                                                   key_file, '*.csv'))
    year_interest = key_file.split('_')[-1]
    date_range = pd.date_range('{}-01-01'.format(year_interest), '{}-12-31'.format(str(int(year_interest))), freq= '1D')

    file_cropsar_ID = [item for item in Cropsar_files_dataset if ID in os.path.split(item)[-1]]
    df_cropsar_ID = pd.read_csv(file_cropsar_ID[0], index_col = 0)
    df_cropsar_ID.index = pd.to_datetime(df_cropsar_ID.index, format= '%Y-%m-%d')
    df_cropsar_ID = df_cropsar_ID.reindex(date_range)
    return df_cropsar_ID

def prepare_SOS_data(indir_satellite_data, ID, key_file, year_interest):
    """
    Function in which the SOS data will be loaded for the field of interest

    :param indir_satellite_data: the basefolder needed to locate all the downloaded TS files
    :param ID:  the ID of the field needed to locate the appropriate file
    :param key_file: the key name of the dataset for which the extractions were done, needed to locate the folder location
    :param year_interest: the year for which the SOS data should be loaded
    :return: a list of SOS values for that field in that specific year
    """
    SOS_files = glob.glob(os.path.join(indir_satellite_data, 'Phenology', key_file, '*.csv'))
    df_SOS = pd.concat((pd.read_csv(f, index_col = 0)for f in SOS_files), axis = 1)
    df_SOS_ID = df_SOS[[ID]].T
    df_SOS_ID = df_SOS_ID[['s1_SOSD_{}'.format(str(year_interest)),'s2_SOSD_{}'.format(str(year_interest))]]
    SOS_values = list(df_SOS_ID.values[0])
    SOS_values = [item for item in SOS_values if not np.isnan(item)]
    return SOS_values


def savgol_interpol(df, var_interpol, window_size = 25):
    """
    Function on which the savitsky golay filter will be applied to covert the TS to daily scale
    :param df: the datadrame containing the TS of the variable (see param 'var_interpol') that should be interpolated
    :param var_interpol: the name of the variable that should be interpolated
    :param window_size: the window size that should be used to do the interpolation
    :return: a dataframe with the variable that is interpolated with the savitsky golay filter
    """
    df.index.name = 'date'
    xr_df = df.to_xarray()

    linear_interpolated = xr_df[var_interpol].interpolate_na(
        dim='date', method='linear').values #, fill_value="extrapolate"
    xr_df[var_interpol].values = savgol_filter(
        linear_interpolated, axis=0,
        window_length= window_size,
        polyorder=3)
    df_interp = pd.DataFrame(data  = xr_df[var_interpol].values, columns = ['{}_smooth'.format(var_interpol)], index = df.index)
    return df_interp


def filter_df_period(df, end_date_period, days_period):
    """
    Function for which the dataframe will be clipped for a specific period
    :param df: dataframe with TS of a metric
    :param end_date_period: the end of the period of interest in the TS
    :param days_period: the amount of days before the end on which should be filtered
    :return: a dataframe clipped to the period of interest before the end date
    """
    period_days_before = end_date_period - datetime.timedelta(days_period)
    df_filter = df[period_days_before.to_datetime64(): end_date_period.to_datetime64()]
    return df_filter

def find_emergence_date_TIMESAT_method(df_ascending, df_descending, SOS, days_period_interest, offset_min_VH = 7, max_dif_orbit_emerg = 14):
    """
    Function used to detected the emergence date by using a priori knowledge from the SOS detected in TIMESAT

    :param df_ascending: the dataframe containing the TS of the ascending orbit
    :param df_descending: the dataframe containing the TS of the descending orbit
    :param SOS: the SOS data for which the corresponding emergence should be defined
    :param days_period_interest: the amount of days before the SOS that can be potential the emergence period
    :param offset_min_VH: the amount of days after the minimum value is considered to be the emergence date
    :param max_dif_orbit_emerg: the maximum difference allowed in emergence prediction between orbit passes.
           If too large, only the orbit with emergence prediction closest to SOS is considered
    :return: emergence date corresponding to the SOS
    """
    df_ascending_filtered = filter_df_period(df_ascending, SOS,days_period_interest)
    df_descending_filtered = filter_df_period(df_descending, SOS,days_period_interest)
    try:
        ascending_emergence = df_ascending_filtered.idxmin().values[0] + np.timedelta64(offset_min_VH,'D')
        descending_emergence = df_descending_filtered.idxmin().values[0] + np.timedelta64(offset_min_VH,'D')
    except:
        print('COULD NOT ESTIMATE EMERGENCE DATA')
        emergence_date = None
        return emergence_date

    diff_emerg = int((ascending_emergence- descending_emergence).astype('timedelta64[D]')/np.timedelta64(1,'D'))
    if diff_emerg > max_dif_orbit_emerg:
        #take the emergence date closest to SOS estimate
        emergence_date = pd.Timestamp(max(ascending_emergence, descending_emergence))
    else:
        ascending_emergence_T_stamp  = pd.Timestamp(ascending_emergence)
        descending_emergence_T_stamp = pd.Timestamp(descending_emergence)
        emergence_date = ascending_emergence_T_stamp + (descending_emergence_T_stamp-ascending_emergence_T_stamp)/2

    emergence_date = emergence_date.date()
    return emergence_date
def detect_slope_peak(df):
    """
    Function used to detect the dates at which a peak in the TS could be observed
    :param df: dataframe with the TS of a specific orbit
    :return: the same dataframe but with indication at which dates a peak in the TS is reached
    """
    df['slope_peak'] = np.sign(df['slope']).diff().eq(-2).shift(-1) #last observation can't be used to find max
    df['slope_peak'] = df['slope_peak'].fillna(False)
    return df

def detect_slope_dip(df):
    """
    Function used to detect the dates at which a dip in the TS could be observed

    :param df: dataframe with the TS of a specific orbit
    :return: the same dataframe but with indication at which dates a dip in the TS is reached
    """
    df['slope_dip'] = np.sign(df['slope']).diff().eq(2).shift(-1) #last observation can't be used to find min
    df['slope_dip'] = df['slope_dip'].fillna(False)
    return df


def check_valley_peaks(peak_dates, df, dip_thr, column_name):
    """
    Function used to check if there is a deep valley between the peak and the earlier peak,
    otherwise this peak could not be used to identify an emergence date in the valley
    :param peak_dates: list of peak dates in the TS
    :param df: dataframe which contains the actual TS
    :param dip_thr: the minimum percentile value that is considered as a dip in the TS
    :param column_name: the column which should be used to detect the dips
    :return: list of peak dates filtered for the deepness of the valley in-between
    """
    peak_dates_filtered = []
    peak_dates_filtered.append(peak_dates[0])
    for ix in range(len(peak_dates)-1):
        df_between_consec_max = df[peak_dates[ix]: peak_dates[ix+1]]
        #check if valley in-between is deep enough -> otherwise remove next max date
        if not df_between_consec_max[df_between_consec_max[column_name[0]]< dip_thr].empty:
            peak_dates_filtered.append(peak_dates[ix+1])
    return peak_dates_filtered

def check_distance_peaks(peak_dates, min_days_local_max = 45):
    """
    Function used to filter the peaks by looking to it's distance from the previous peak
    :param peak_dates: list of peak dates in the TS
    :param min_days_local_max: the minimul distance in days that the peaks should be seperated from each other
    :return: filtered list of peaks
    """
    peak_dates_filtered = []
    peak_dates_filtered.append(peak_dates[0]) # the first peak can't be checked for distance with the previous one
    for px in range(len(peak_dates)-1):
        days_diff = np.abs((peak_dates[px] - peak_dates[px+1])/np.timedelta64(1,'D'))
        if days_diff > min_days_local_max:
            peak_dates_filtered.append(peak_dates[px+1])
    return peak_dates_filtered

def find_unique_max_season_dates(df,thr_max_find = 70, thr_min_dip_inbetween = 20):
    """
    Function used to detect all the peak dates in the TS which could be considered as a maximum of the season.

    :param df: dataframe containing the TS of the variable and for a specific orbit for which the emergence will be estimated
    :param thr_max_find: the percentile which marks the lower threshold for detecting peaks in the TS
    :param thr_min_dip_inbetween: The minimum percentile for which the TS should go under in order to decide if the valley before the max is deep enough
    :return: all the dates with peaks in the TS
    """
    column_name = list(df.columns)
    #the slope will be needed to define the peaks in the TS
    df['slope'] = df[column_name[0]].diff()

    #find cells that shift from rising to decrease of slope (potential max)
    df = detect_slope_peak(df)

    #find the minimum value that will be considered as a peak
    define_max_thr  = np.nanpercentile(df[column_name[0]], thr_max_find)
    peak_dates = list(df.loc[((df['slope_peak'] == True)
                              & (df[column_name[0]] > define_max_thr))].index.values)
    #needed to seperate seasons from each other
    dip_thr = np.nanpercentile(df[column_name[0]], thr_min_dip_inbetween)

    #the first one can't be checked if there is a deep enough valley before
    peak_dates_filtered_dips = check_valley_peaks(peak_dates, df, dip_thr, column_name)

    #now check that the peaks are enough seperated from each other
    peak_dates_filtered_distance = check_distance_peaks(peak_dates_filtered_dips)

    #now remove the first local max if there is no local min before
    df = detect_slope_dip(df)
    df_before_first_peak = df[df.index[0]: peak_dates_filtered_distance[0]]
    if df_before_first_peak.loc[df_before_first_peak['slope_dip'] == True].empty: #remove first peak
        peak_dates_filtered_distance = peak_dates_filtered_distance[1:]

    # if len(peak_dates_filtered_distance) > max_seasons: #only store x values each time:
    #     #take the highest peaks for emergence detection
    #     peak_dates_final = list(df[df.index.isin(peak_dates_filtered_distance)]\
    #                         .sort_values(by = [column_name[0]], ascending= False).index.values[0:max_seasons])
    # else:
    #     peak_dates_final = peak_dates_filtered_distance

    peak_dates_final = peak_dates_filtered_distance
    peak_dates_final.sort()
    return peak_dates_final

def check_for_local_dip(df,df_dip_dates, min_diff_peak_dip = 20, min_diff_dips_same_season = 50, min_diff_dips = -2):
    if df_dip_dates.shape[0]>2:
        #check if previous dip is not higher than first dip before peak
        if df_dip_dates[df_dip_dates.columns.values[0]].values[-2]-df_dip_dates[df_dip_dates.columns.values[0]].values[-1] >0:
            df_dip_dates = df_dip_dates.drop(df_dip_dates.index.values[-2])


    closest_peak = df[:df_dip_dates.index.values[-1]].loc[(df[:df_dip_dates.index.values[-1]].slope_peak == True)]
    closest_peak_date = closest_peak.index.values[-1]
    diff_closest_peak = (df_dip_dates.index.values[-1]-closest_peak_date)/ np.timedelta64(1,'D')
    diff_dip_closest_peak = (df_dip_dates.index.values[-1]- df_dip_dates.index.values[-2])/ np.timedelta64(1,'D')
    if diff_dip_closest_peak < min_diff_dips_same_season and diff_closest_peak < min_diff_peak_dip :
        diff_dips_dB = df_dip_dates[df_dip_dates.columns.values[0]].values[-2] - df_dip_dates[df_dip_dates.columns.values[0]].values[-1]
        if diff_dips_dB < min_diff_dips:
            df_dip_dates = df_dip_dates.drop(df_dip_dates.index.values[-1])
    return df_dip_dates


def find_left_min(peak_dates, df, thr_min_dip = 15, offset_dip = 7):
    """
    Function used to find the date of the left minimum  + offset (considered as potential emergence)
    before the peak of the season.
    :param peak_dates: list of peak dates in the TS
    :param df: dataframe containing the TS
    :param thr_min_dip: the minimum percentile of the TS for which values below could be considered as minimum
    :param offset_dip: The amount of days after the minimum that are considered as the event that should be detected
    :return:
    """
    dip_dates  = []
    columns = list(df.columns)
    min_dip_value = np.nanpercentile(df[columns[0]], thr_min_dip)
    for date in peak_dates:
        #take the closest dip before the max
        df_dip_date = df[:date].loc[((df[:date].slope_dip == True)&
                                     (df[:date][columns[0]]< min_dip_value))]

        ### additional check for local minimum -> For the moment skip this step
        # if df_dip_date.shape[0]>1:
        #     df_dip_date = check_for_local_dip(df,df_dip_date)

        idx_peak = peak_dates.index(date)

        if not df_dip_date.empty:
            dip_date = df_dip_date.index.values[-1]
            if idx_peak >0 and dip_date < peak_dates[idx_peak-1]:
                continue
            dip_date = dip_date + np.timedelta64(offset_dip,'D')
            dip_dates.append(dip_date)
    return list(set(dip_dates))
def remove_too_close_emerg_pred(lst_emerg_estimates_both_orbits, min_diff_pred = 45):
    """

    :param lst_emerg_estimates_both_orbits: list of proposed emergence dates based on the combination of two orbits
    :param min_diff_pred: the minimum difference in days that is required to define emergence dates to belong to a different season
    :return: updated list of emergence dates for which at least min_diff_pred is guaranteed between the predictions
    """
    diff_days_emergence_est = [j-i for i, j in zip(lst_emerg_estimates_both_orbits[:-1], lst_emerg_estimates_both_orbits[1:])]
    ### find which difference is too close
    diffs_days_too_close = [item for item in diff_days_emergence_est if item.days <min_diff_pred]
    lst_remove = []
    ## now start searching which emergence dates are too close
    for diff_day in diffs_days_too_close:
        index_emerg_dates_too_close = diff_days_emergence_est.index(diff_day)
        emerg_dates_too_close = lst_emerg_estimates_both_orbits[index_emerg_dates_too_close:index_emerg_dates_too_close+2]
        emerg_dates_too_close = sorted(emerg_dates_too_close)
        # the first emergence date out of the two is considered as the most valid one
        emerg_remove = emerg_dates_too_close[-1]
        lst_remove.append(emerg_remove)
    lst_emerg_estimates_both_orbits_updated = [item for item in lst_emerg_estimates_both_orbits if not item in lst_remove]
    return lst_emerg_estimates_both_orbits_updated


def calc_final_emerg_both_orbits(emerg_dates_asc, emerg_dates_desc, max_diff_emerg_orbits = 30, max_seasons =2):
    """
    Function used to combine the potential emergence dates from both orbits to get a final emergence prediction
    :param emerg_dates_asc: the potential emergence dates based on the ascending orbit
    :param emerg_dates_desc: the potential emergence dates based on the descending orbit
    :param max_diff_emerg_orbits: the max difference the potential emergence data may be seperated between the orbits
    :param max_seasons: the max amount of seasons per year that are taken into account for the emergence detection.
    This condition is only used when there are less then two emergence dates detected by combining both orbits
    :return: list of emergence dates based on the combination of both orbits and
    another list of emergence dates for which no match with the other orbit pass could be found
    """
    lst_emerg_both_orbits = []
    lst_emerg_single_orbit = []
    lst_emerg_desc_match = []
    lst_emerg_asc_match = []
    #align emergence prediction for both orbits
    for x in range(len(emerg_dates_asc)):
        find_closest = [np.abs((emerg_dates_asc[x]- emerg_dates_desc[p])/np.timedelta64(1,'D')) for p in range(len(emerg_dates_desc))]
        if min(find_closest)<= max_diff_emerg_orbits:
            emerg_date_desc_match = emerg_dates_desc[find_closest.index(min(find_closest))]
            lst_emerg_desc_match.append(emerg_date_desc_match)
            emerg_date_desc_match = pd.Timestamp(emerg_date_desc_match)
            emerg_date_asc_match  = emerg_dates_asc[x]
            lst_emerg_asc_match.append(emerg_date_asc_match)
            emerg_date_asc_match = pd.Timestamp(emerg_date_asc_match)
            avg_emergence_date = emerg_date_asc_match + (emerg_date_desc_match-emerg_date_asc_match)/2
            avg_emergence_date = avg_emergence_date.date()
            lst_emerg_both_orbits.append(avg_emergence_date)

    ### check additionally if the provided emergence dates are at least x days separated from each other
    if len(lst_emerg_both_orbits) >1:
        lst_emerg_both_orbits = remove_too_close_emerg_pred(lst_emerg_both_orbits)



    if len(lst_emerg_both_orbits) < max_seasons:
        emerg_dates_asc_no_match = [pd.Timestamp(item) for item in emerg_dates_asc if item not in lst_emerg_asc_match]
        emerg_dates_desc_no_match = [pd.Timestamp(item) for item in emerg_dates_desc if item not in lst_emerg_desc_match]
        #TODO COULD ADD CONDITION TO ONLY SELECT HIGHEST PEAK, BUT FOR NOW ONE RANDOMLY IS SELECTED
        if emerg_dates_desc_no_match:
            lst_emerg_single_orbit.extend(emerg_dates_desc_no_match[0:max_seasons-len(lst_emerg_both_orbits)])
        if emerg_dates_asc_no_match:
            lst_emerg_single_orbit.extend(emerg_dates_asc_no_match[0:max_seasons-len(lst_emerg_both_orbits)])

    return lst_emerg_both_orbits, lst_emerg_single_orbit






def find_emergence_date_S1_only(df_ascending, df_descending,
                                offset_dip = 7):
    """
    Function for which all the possible emergence dates within a TS in the dataframe will be determined
    without a priori knowledge for which period the emergence could take place.

    :param df_ascending: dataframe containing the preprocessed TS (daily values) of the ascending orbit
    :param df_descending: dataframe containing the preprocessed TS (daily values) of the descending orbit
    :param offset_dip: the amount of days that the emergence date is estimated from the minimum
    :return: list of emergence dates from both orbits and list if emergence data was only for a single orbit
    """
    peak_dates_asc  = find_unique_max_season_dates(df_ascending)
    peak_dates_desc = find_unique_max_season_dates(df_descending)
    #TODO find solution to better align the detection for both orbits



    # find left local min for each max
    emerg_dates_asc = find_left_min(peak_dates_asc, df_ascending, offset_dip = offset_dip)
    emerg_dates_desc = find_left_min(peak_dates_desc, df_descending, offset_dip = offset_dip)

    #now check that the distance in minima is not too big between both orbits


    #write function to define the emergence dates closest to each other for both orbit passes
    if not emerg_dates_desc or not emerg_dates_asc:
        return [], []
    else:
        emerg_dates_both_orbits, emerg_dates_single_orbit = calc_final_emerg_both_orbits(emerg_dates_asc, emerg_dates_desc)

    return emerg_dates_both_orbits, emerg_dates_single_orbit
