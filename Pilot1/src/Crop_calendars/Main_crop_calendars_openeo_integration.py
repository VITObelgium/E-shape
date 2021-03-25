import os
from Crop_calendars.Crop_calendars_openeo_integration import Cropcalendars
import json
def main():
    ######## DEFAULT PARAMETERS ##########
    metrics_order = ['sigma_ascending_VH', 'sigma_ascending_VV','sigma_ascending_angle','sigma_descending_VH', 'sigma_descending_VV','sigma_descending_angle', 'fAPAR']
    generator = Cropcalendars(fAPAR_rescale_Openeo= 0.005,
                              path_harvest_model= r'/data/users/Public/bontek/e_shape/model/model_update1.0_iteration24.h5',
                              VH_VV_range_normalization= [-13, -3.5], fAPAR_range_normalization= [0,1], metrics_order = metrics_order,
                              window_values = 5, thr_detection = 0.75, crop_calendar_event = 'Harvest',
                              metrics_crop_event = ['cropSAR', 'VH_VV_{}'],
                              index_window_above_thr = 2, max_gap_prediction = 24,
                              shub = True)

    """ INFO DEFAULT METRICS
    fAPAR_rescale_Openeo: rescaling factor of fapar in case the data is retrieved from SHUB
    path_harvest_model: location of the stored NN model
    VH_VV_range_normalization: ranges used to normalize the VH/VV to values between -1 and 1
    fAPAR_range_normalization:   ranges used to normalize the fapar to values between -1 and 1
    metrics_order: the order of the timeseries metrics stored in the timeseries dataframe
    window_values: define the amount of S1 coverages within the window for extraction
    thr_detection: threshold for crop event detection
    crop_calendar_event: the name of crop cale
    metrics_crop_event: the metrics used to determine the crop calendar event
    index_window_above_thr: the x-th position that the harvest probability detection exceeds the probability threshold ('thr_detection')
    max_gap_prediction:  the max amount of days that is allowed between succeeding harvest probabilities 
                         above the probability threshold ('thr_detection'). If the gap is larger this 
                         harvest prediction is ignored for the 'index_window_above_thr'

    shub: indicating of the satellite data should be retrieved from shub (True) or Terrascope (False)
    
    """



    #### USER SPECIFIC PARARMETERS
    # the directory of the file (geojson format) which is going
    # to be used to determine the specific crop calendar event
    gjson_path =r'' #Path("../../Tests/Cropcalendars/EX_files/WIG_harvest_detection_fields.geojson")
    ## define the time period for extracting the time series data
    start = '2019-01-01'
    end = '2019-07-31'
    # the folder in which you want to store the output result
    outdir = r''

    #the name of the output file containing the crop calendar
    #info for the fields
    outname = r'Test_fields_after_RO_selection_and_orbit_direction_SHUB.json'#r'Extract_LPIS_test.json'


    ###### INITIATE THE CLASS AND RUN THE CROP CALENDAR MODEL
    # The output contains an updated geojson file with
    # in its properties the derived crop calendar events
    gj_cropcalendars_info = generator.generate_cropcalendars_local(start = start, end = end, gjson_path = gjson_path) # returns the geometry collection with as attribute the crop calendars per field ID
    ### The output file location (json format) which you want to use to store the result
    with open(os.path.join(outdir, outname),"w") as file:
        file.write(json.dumps(gj_cropcalendars_info))


if __name__ == '__main__':

    main()






