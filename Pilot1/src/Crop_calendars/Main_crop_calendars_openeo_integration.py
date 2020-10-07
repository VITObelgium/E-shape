from pathlib import Path

from Crop_calendars.Crop_calendars_openeo_integration import Cropcalendars
import json
def main():
    ######## DEFAULT PARAMETERS ##########
    metrics_order =  ['sigma_ascending_VH', 'sigma_ascending_VV','sigma_ascending_angle','sigma_descending_VH', 'sigma_descending_VV','sigma_descending_angle', 'fAPAR']  # The index position of the metrics returned from the OpenEO datacube
    generator = Cropcalendars(fAPAR_rescale_Openeo= 0.005, coherence_rescale_Openeo= 0.004,
                              path_harvest_model= r"../../Tests/Cropcalendars/Model/model_update1.0_iteration24.h5",
                              VH_VV_range_normalization= [-13, -3.5], fAPAR_range_normalization= [0,1], metrics_order = metrics_order)

    #### USER SPECIFIC PARARMETERS
    gjson_path = Path("../../Tests/Cropcalendars/EX_files/WIG_harvest_detection_fields.geojson")
    ## define the period (year) for extracting the time series data
    year_of_interest = 2019
    start = '{}-01-01'.format(str(year_of_interest))
    end = '{}-12-31'.format(str(year_of_interest))

    ##### CROP CALENDAR EVENT SPECIFIC PARAMETERS FOR THE EVENT THAT NEEDS TO BE DETERMINED
    window_values = 5 # define the amount of S1 coverages within the window for extraction
    thr_detection = 0.75 # threshold for crop event detection
    index_window_above_thr = 2 # the index position above the threshold that will be used to define the crop event date
    crop_calendar_event = 'Harvest'
    metrics_crop_event = ['cropSAR', 'VH_VV_{}'] # the metrics used to determine the crop calendar event
    gj_cropcalendars_info = generator.generate_cropcalendars(start = start, end = end, gjson_path = gjson_path, window_values= window_values, thr_detection= thr_detection,
                                                             crop_calendar_event= crop_calendar_event, metrics_crop_event = metrics_crop_event,
                                                             index_window_above_thr = index_window_above_thr) # returns the geometry collection with as attribute the crop calendars per field ID

    ### The output file (json format) which you want to use to store the result
    with open(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\Extract_LPIS_test.json","w") as file:
        file.write(json.dumps(gj_cropcalendars_info))


if __name__ == '__main__':

    main()






