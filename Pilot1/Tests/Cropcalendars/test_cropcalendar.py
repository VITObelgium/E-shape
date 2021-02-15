
### Unit tests for cropcalendar, use PyTest to run.
###
from pathlib import Path

import pandas as pd
import pytest
import openeo
import shapely
import json
from shapely.geometry import mapping
from Crop_calendars.Crop_calendars_openeo_integration import Cropcalendars
from openeo.rest.conversions import timeseries_json_to_pandas
from openeo.rest.datacube import DataCube


@pytest.fixture
def connection():
    return openeo.connect("https://openeo-dev.vito.be").authenticate_basic()

@pytest.fixture
def crop_calendars(connection):
    metrics_order = ['sigma_ascending_VH', 'sigma_ascending_VV', 'sigma_ascending_angle', 'sigma_descending_VH',
                     'sigma_descending_VV', 'sigma_descending_angle',
                     'fAPAR']  # The index position of the metrics returned from the OpenEO datacube
    generator = Cropcalendars(fAPAR_rescale_Openeo=0.005, coherence_rescale_Openeo=0.004,
                              path_harvest_model=r'/data/users/Public/bontek/e_shape/model/model_update1.0_iteration24.h5',
                              VH_VV_range_normalization=[-13, -3.5], fAPAR_range_normalization=[0, 1],
                              metrics_order=metrics_order,connection=connection)
    return generator

start = '2019-01-01'
end = '2019-12-31'
gjson_path = Path(__file__).parent / "EX_files" /"WIG_harvest_detection_fields.geojson"

@pytest.fixture()
def geometry():
    gj, polygons_inward_buffered = Cropcalendars.load_geometry(gjson_path)
    geo = shapely.geometry.GeometryCollection(
        [shapely.geometry.shape(feature).buffer(0) for feature in polygons_inward_buffered])
    return geo

def write_json(dict, file):
    with(open(file, 'w+')) as out:
        json.dump(dict, out)

def test_prepare_geometry(geometry):
    write_json(mapping(geometry),"buffered.geojson")



def test_retrieve_inputs(connection,crop_calendars,geometry):
    bands_ts: DataCube = crop_calendars.get_bands()
    #bands_ts.to_graphviz().render("graph.png")
    bands_ts.filter_temporal(start, end).polygonal_mean_timeseries(geometry).download("timeseries2.json",format="JSON")

def test_retrieve_inputs_netcdf(connection,crop_calendars,geometry):
    bands_ts: DataCube = crop_calendars.get_bands()
    #bands_ts.to_graphviz().render("graph.png")
    bands_ts.filter_temporal('2019-05-01', '2019-06-01').filter_bbox(geometry).download("timeseries.nc",format="NetCDF")

def test_udf_local():
    unique_ids_fields = []
    dict_ascending_orbits_field = dict()
    dict_descending_orbits_field = dict()

    with(open("timeseries.json","r+")) as f:
        ts_dict = json.load(f)
        df_metrics = timeseries_json_to_pandas(ts_dict)
        df_metrics.index = pd.to_datetime(df_metrics.index)

        # use the UDF to determine the crop calendars for the fields in the geometrycollection
        # from .crop_calendar_udf import udf_cropcalendars
    from Crop_calendars.crop_calendar_local import udf_cropcalendars_local
    # crop_calendars = udf_cropcalendars(df_metrics, unique_ids_fields)


    crop_calendars_df = udf_cropcalendars_local(ts_dict, unique_ids_fields, dict_ascending_orbits_field,
                                                dict_descending_orbits_field)

