### In this script the UDP for the harvest detector will be called
import openeo
import json
import geojson
import shapely
import os
connection = openeo.connect('https://openeo.vito.be/openeo/1.0.0') \
    .authenticate_basic('bontek', 'bontek123')
#gjson_path = '/data/users/Public/nielsh/WIG_harvest_detection_fields.geojson'

""" USER-DEFINED PARAMETERS FOR RUNNING THE HARVEST PREDICTOR"""
time_range = '2019-01-01', '2019-12-31' #startdate, enddate for the harvest prediction
gjson_path = r"S:\eshape\Pilot 1\results\Harvest_date\Code_testing\Field_BE_test\Fields\Field_BE.geojson"  #path at which the geojson file is located # r"/data/users/Public/nielsh/WIG_harvest_detection_fields.geojson"
outdir = r'S:\eshape\Pilot 1\results\Harvest_date\Code_testing\Field_BE_test\Output' # the directory to which the output should be written
outname = r'Field_BE_harvest_prediction_20190101_20191231' # the name of the output file containing the predicted harvest date for each field


""" LOAD THE GEOJSON FILE LOCALLY IF IT IS STORED ON THE PUBLIC DRIVE OF THE TERRASCOPE VM
    OTHERWISE LOAD THE GEOJSON FILE LOCALLLY"""
if not "data" in gjson_path and not "Public" in gjson_path:
    with open(gjson_path) as f:
        gjson_path = geojson.load(f)
      #gjson_path = shapely.geometry.GeometryCollection([shapely.geometry.shape(feature.geometry).buffer(0) for feature in gj.features])


""" CALL AND RUN THE PROCESS FOR PREDICTING HARVEST DATES"""
crop_calendar = connection.datacube_from_process('CropCalendar', namespace='nextland', time_range=time_range, gjson_path=gjson_path)
result = crop_calendar.send_job().start_and_wait().get_result().load_json()

""" STORE THE RESULT IN THE JSON FILE"""
with open(os.path.join(outdir, outname + '.json'), 'w') as f:
    f.write(json.dumps(result))