from pathlib import Path

from Crop_calendars.Crop_calendars_openeo_integration import Cropcalendars
import json
def main():
    ######## DEFAULT PARAMETERS ##########
    metrics_order = ['gamma_VH', 'gamma_VV','sigma_ascending_VH', 'sigma_ascending_VV','sigma_angle','sigma_descending_VH', 'sigma_descending_VV','sigma_descending_angle', 'fAPAR']  # The index position of the metrics returned from the OpenEO datacube
    generator = Cropcalendars(fAPAR_rescale_Openeo= 0.005, coherence_rescale_Openeo= 0.004,
                              path_harvest_model= r"C:\Users\bontek\git\e-shape\Pilot1\Tests\Cropcalendars\Model\model_update1.0_iteration24.h5",
                              VH_VV_range_normalization= [-13, -3.5], fAPAR_range_normalization= [0,1], metrics_order = metrics_order)

    #### USER SPECIFIC PARARMETERS
    gjson_path = Path("../../Tests/Cropcalendars/EX_files/Field_test.geojson")
    ## define the period (year) for extracting the time series data
    year_of_interest = 2019
    start = '{}-01-01'.format(str(year_of_interest))
    end = '{}-12-31'.format(str(year_of_interest))

    ##### CROP CALENDAR EVENT SPECIFIC PARAMETERS FOR THE EVENT THAT NEEDS TO BE DETERMINED
    window_values = 5 # define the amount of S1 coverages within the window for extraction
    thr_detection = 0.8 # threshold for crop event detection
    crop_calendar_event = 'Harvest'
    metrics_crop_event = ['fAPAR', 'VH_VV_{}'] # the metrics used to determine the crop calendar event
    gj_cropcalendars_info = generator.generate_cropcalendars(start = start, end = end, gjson_path = gjson_path, window_values= window_values, thr_detection= thr_detection,
                                                             crop_calendar_event= crop_calendar_event, metrics_crop_event = metrics_crop_event) # returns the geometry collection with as attribute the crop calendars per field ID
    with open(r"S:\eshape\Pilot 1\NB_Jeroen_OpenEO\eshape\output_test\TAP_fields_datacube_metrics_test_crop_calendar_output_GEE_update.json","w") as file:
        file.write(json.dumps(gj_cropcalendars_info))


if __name__ == '__main__':

    main()






