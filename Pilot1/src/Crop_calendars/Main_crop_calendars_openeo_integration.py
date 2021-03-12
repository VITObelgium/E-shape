import os
from Crop_calendars.Crop_calendars_openeo_integration import Cropcalendars
import json
def main():
    ######## DEFAULT PARAMETERS ##########
    metrics_order = ['sigma_ascending_VH', 'sigma_ascending_VV','sigma_ascending_angle','sigma_descending_VH', 'sigma_descending_VV','sigma_descending_angle', 'fAPAR']  # The index position of the metrics returned from the OpenEO datacube
    generator = Cropcalendars(fAPAR_rescale_Openeo= 0.005, coherence_rescale_Openeo= 0.004,
                              path_harvest_model= r'/data/users/Public/bontek/e_shape/model/model_update1.0_iteration24.h5',
                              VH_VV_range_normalization= [-13, -3.5], fAPAR_range_normalization= [0,1], metrics_order = metrics_order)

    #### USER SPECIFIC PARARMETERS
    # the directory of the file (geojson format) which is going
    # to be used to determine the specific crop calendar event
    gjson_path = r"S:\eshape\tmp\harvest_detector\Fields_US\Field_test.geojson" #Path("../../Tests/Cropcalendars/EX_files/WIG_harvest_detection_fields.geojson")
    ## define the time period for extracting the time series data
    start = '2019-01-01'#'2020-05-01'
    end = '2021-02-02'#'2020-10-31'
    # the folder in which you want to store the output result
    outdir = r'S:\eshape\tmp\harvest_detector'

    #the name of the output file containing the crop calendar
    #info for the fields

    outname = r'Test_fields.json'#r'Extract_LPIS_test.json'

    ##### CROP CALENDAR EVENT SPECIFIC PARAMETERS FOR THE EVENT THAT NEEDS TO BE DETERMINED
    window_values = 5 # define the amount of S1 coverages within the window for extraction
    thr_detection = 0.75 # threshold for crop event detection
    index_window_above_thr = 2 # the index position above the threshold that will be used to define the crop event date
    max_gap_prediction = 24  # the max amount of days that are allowed between harvest detection predictions
    crop_calendar_event = 'Harvest'
    metrics_crop_event = ['cropSAR', 'VH_VV_{}'] # the metrics used to determine the crop calendar event
    shub = True

    ###### INITIATE THE CLASS AND RUN THE CROP CALENDAR MODEL
    # The output contains an updated geojson file with
    # in its properties the derived crop calendar events
    gj_cropcalendars_info = generator.generate_cropcalendars(start = start, end = end, gjson_path = gjson_path, window_values= window_values, thr_detection= thr_detection,
                                                             crop_calendar_event= crop_calendar_event, metrics_crop_event = metrics_crop_event,
                                                             index_window_above_thr = index_window_above_thr, shub = shub, max_gap_prediction = max_gap_prediction) # returns the geometry collection with as attribute the crop calendars per field ID
    ### The output file location (json format) which you want to use to store the result
    with open(os.path.join(outdir, outname),"w") as file:
        file.write(json.dumps(gj_cropcalendars_info))


if __name__ == '__main__':

    main()






