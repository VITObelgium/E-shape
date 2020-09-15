from Pilot1.src.Crop_calendars.Main_functions import Openeo_extraction_S2_allbands
from pathlib import Path
from openeo.rest.conversions import timeseries_json_to_pandas
import geojson
import shapely
import json
import pandas as pd
import numpy
import geopandas as gpd
run_local = False
start = '2019-01-01'
end = '2020-06-30'
outfolder = r'S:\eshape\Pilot 1\results\Bare_soil_detection\data\S2'
#shp = gpd.read_file(r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.shp")
shp = gpd.read_file(r'/data/users/Public/bontek/e_shape/data/2019_TAP_monitoring_experiment.shp')

#### S2 extract
Openeo_extraction_S2_allbands(start,end, shp, outfolder, batch= True)




