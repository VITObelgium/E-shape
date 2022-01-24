from Crop_calendars.Emergence.preprocess_TS_emergence import preprocess_S1_data, \
    savgol_interpol, prepare_SOS_data,find_emergence_date_TIMESAT_method, \
    find_emergence_date_S1_only
import geopandas as gpd
import pandas as pd




indir_satellite_data = r'' #folder in which all the data needed will be stored
ID_identifier = 'id' #the attribute column from which the unique IDS can be taken

## define the window size for smoothing the data
interp_window_size = 35 # the size of the window used in the SAVITSKY GOLAY filter
days_period_search_SOS = 60 #the amount of days before the SOS data there are used to further fine-tune emergence detection
TIMESAT_method = False

##define the dataset for which the emergence will be detected
dict_shp_files = {} #-> should be structured like this: {'key_name of file': shapefile directory of file}



for key_file in dict_shp_files.keys():
    #load the dataset
    shp_dataset = gpd.read_file(dict_shp_files.get(key_file))
    IDS_fields =  shp_dataset[ID_identifier].to_list()
    year_interest = key_file.split('_')[-1]


    for ID in IDS_fields:

        #load the already downloaded Sentinel-1 backscatter intensity data

        df_ascending_sigma_ID, df_descending_sigma_ID = preprocess_S1_data(indir_satellite_data, ID,
                                                                           key_file)
        if df_ascending_sigma_ID is None or df_descending_sigma_ID is None:
            print('EMPTY ORBIT FILE FOR {}'.format(ID))
            continue

        #apply some gap-filling on the Sentinel-1 data to remove noise
        df_ascending_sigma_ID_interp = savgol_interpol(df_ascending_sigma_ID, 'VH', window_size = interp_window_size)
        df_descending_sigma_ID_interp = savgol_interpol(df_descending_sigma_ID, 'VH', window_size = interp_window_size)

       ## if the TIMESAT method is used the SOS values are used to fine-tune it to an emergence date
        if TIMESAT_method:
            emergence_result = []
            SOS_values = prepare_SOS_data(indir_satellite_data,  ID, key_file, year_interest)
            ##now that we have phenology information -> start searching the actual emergence
            for SOS in SOS_values:
                SOS_datetime = pd.to_datetime(SOS, format = '%y%j')
                print('THE ESTIMATES SOS FOR {} IS {}'.format(ID,SOS_datetime.date()))


                #Apply the emergence detector based on TIMESAT
                emergence_date = find_emergence_date_TIMESAT_method(df_ascending_sigma_ID_interp,
                                                                    df_descending_sigma_ID_interp,
                                                                    SOS_datetime,
                                                                    days_period_search_SOS)
                emergence_result.append(emergence_date)

        else:
            ##method without a priori knowledge on the potential moment of emergence
            emergence_result, emergence_result_single_orbit = find_emergence_date_S1_only(df_ascending_sigma_ID_interp, df_descending_sigma_ID_interp)
        print(emergence_result)








