#import openeo
#from openeo.internal.graph_building import PGNode
import geopandas as gpd
#from openeo.rest.conversions import timeseries_json_to_pandas
import ee
import geopandas as gpd
import pandas as pd
import statistics
##### GEE to fine RO of parcels
# Initialize the Earth Engine object, using the authentication credentials.
print('Initializing Earth Engine API...')
ee.Initialize()
start_date = '2019-01-01'
end_date = '2019-12-31'
dict_descending_orbits_field = dict()
dict_ascending_orbits_field = dict()


###############################################################################
# Main script
###############################################################################

# Initiate task list
all_tasks = []

# Import the collections
sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD")
shp_example = gpd.read_file(r"S:\WIG\Validation_parcels\KVE2018n.shp")
coord = [list(shp_example.geometry.exterior[row_id].coords) for row_id in range(shp_example.shape[0])][0]
coord = [list(elem) for elem in coord]

collection = ee.FeatureCollection(
    [ee.Feature(
        ee.Geometry.Polygon(
            [coord
              ]
        ), {'Name': 'Test'}
    )]
)
filter_field = collection.filter(ee.Filter.eq('Name', 'Test'))

try:
    ###############################################################################
    # PROCESSING SENTINEL 1
    ###############################################################################
    dict_metadata_ascending = dict()
    dict_metadata_descending = dict()
    for mode in ['ASCENDING', 'DESCENDING']:
        print('Extracting Sentinel-1 data in %s mode' % (mode))
        # Filter S1 by metadata properties.
        sentinel1_filtered = sentinel1.filterBounds(filter_field.geometry().bounds()).filterDate(start_date, end_date) \
            .filter(ee.Filter.eq('orbitProperties_pass', mode)) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

        sentinel1_collection_contents = ee.List(sentinel1_filtered).getInfo()
        current_nr_files = len(sentinel1_collection_contents['features'])

        print('{} Sentinel-1 images match the request ...'.format(current_nr_files))
        for img_nr in range(current_nr_files):
            current_sentinel_img_id = str(sentinel1_collection_contents['features'][img_nr]['id'])

            # if want to know the incidence angle for the field
            #current_sentinel_img = ee.Image(current_sentinel_img_id)
            #current_sentinel_img.clip(filter_field.geometry()).reduceRegion(ee.Reducer.mean()).getInfo()['angle']

            if 'S1A' in current_sentinel_img_id:
                RO = ((int(current_sentinel_img_id.rsplit('_')[7][1:])-73)%175)+1
            if 'S1B' in current_sentinel_img_id:
                RO = ((int(current_sentinel_img_id.rsplit('_')[7][1:])-27)%175)+1
            if mode  == 'ASCENDING':
                dict_metadata_ascending.update({pd.to_datetime(current_sentinel_img_id.rsplit('_')[5][0:8]): RO})
            if mode == 'DESCENDING':
                dict_metadata_descending.update({pd.to_datetime(current_sentinel_img_id.rsplit('_')[5][0:8]): RO})

except KeyboardInterrupt:
    raise
# TO DO can eventually use the max angle orbit
RO_ascending_selection = statistics.mode(list(dict_metadata_ascending.values()))
RO_descending_selection = statistics.mode(list(dict_metadata_descending.values()))
dict_ascending_orbits_field.update({filter_field: RO_ascending_selection})
dict_descending_orbits_field.update({filter_field : RO_descending_selection})




#### WAY TO FIND RO FROM OPENEO

# connection = openeo.connect('http://openeo-dev.vgt.vito.be/openeo/1.0.0/')
# connection.authenticate_basic('bontek', 'bontek123')
# start = '2018-01-01'
# end = '2018-02-15'
# data = connection.load_collection(
#     "TERRASCOPE_S1_GAMMA0_V1",
#     properties={
#         "relativeOrbitNumber": PGNode(process_id="eq", arguments={"x": {"from_parameter": "value"}, "y": 161})  ### y: RO number
#     }
# )
# shp = gpd.read_file(r"S:\eshape\Pilot 1\data\WIG_data\Field.shp")
# cube = data.filter_temporal(start, end).polygonal_mean_timeseries(shp).execute()
# ts = timeseries_json_to_pandas(cube)


