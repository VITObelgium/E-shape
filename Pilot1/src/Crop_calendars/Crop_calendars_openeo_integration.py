from pathlib import Path
import pandas as pd
import openeo
import shapely
from openeo.rest.conversions import timeseries_json_to_pandas
from Crop_calendars.Terrascope_catalogue_retrieval import OpenSearch
from shapely.geometry.polygon import Polygon
from Crop_calendars.create_mask import create_mask
from Crop_calendars.prepare_geometry import prepare_geometry,remove_small_poly
import geojson
import json
from openeo.processes import eq

# General approach:
#
# first merge all required inputs into a single multiband raster datacube
# compute timeseries for one or more fields, for all bands (one step)
# do postprocessing of the timeseries:
#   compute cropsar based on cleaned fapar + sigma0
#   combine cropsar + S1 to determine cropcalendar
#   return cropcalendar output in your own json format

class Cropcalendars():
    def __init__(self, fAPAR_rescale_Openeo, path_harvest_model,VH_VV_range_normalization, fAPAR_range_normalization,
                 metrics_order, window_values, thr_detection,
                 crop_calendar_event,metrics_crop_event ,max_gap_prediction ,
                 shub, index_window_above_thr,connection = None):

        # crop calendar independant variables
        self.fAPAR_rescale_Openeo = fAPAR_rescale_Openeo
        self.path_harvest_model = path_harvest_model
        self.VH_VV_range_normalization = VH_VV_range_normalization
        self.fAPAR_range_normalization  = fAPAR_range_normalization
        self.metrics_order = metrics_order
        self.window_values = window_values
        self.thr_detection = thr_detection
        self.crop_calendar_event = crop_calendar_event
        self.metrics_crop_event = metrics_crop_event
        self.max_gap_prediction = max_gap_prediction
        self.shub = shub
        self.index_window_above_thr = index_window_above_thr

        # openeo connection
        if(connection == None):

            self._eoconn = openeo\
                .connect('https://openeo-dev.vito.be/openeo/1.0.0')\
                .authenticate_basic('bontek', 'bontek123')
        else:
            self._eoconn = connection
        self._open_search = OpenSearch()

    #####################################################
    ################# FUNCTIONS #########################
    #####################################################

    def get_resource(self,relative_path):
        return str(Path(relative_path))

    def load_udf(self, relative_path):
        with open(self.get_resource(relative_path), 'r+', encoding="utf8") as f:
            return f.read()

    def get_bands(self, shub = False):
        if not shub:
            S2mask = create_mask(self._eoconn)
            fapar = self._eoconn.load_collection('TERRASCOPE_S2_FAPAR_V2', bands=['FAPAR_10M'])

            fapar_masked = fapar.mask(S2mask)

            sigma_ascending = self._eoconn.load_collection('S1_GRD_SIGMA0_ASCENDING', bands=["VH", "VV", "angle"])
            sigma_descending = self._eoconn.load_collection('S1_GRD_SIGMA0_DESCENDING',bands=["VH", "VV", "angle"]).resample_cube_spatial(sigma_ascending)

            fapar_masked = fapar_masked.resample_cube_spatial(sigma_ascending)

            all_bands = sigma_ascending.merge(sigma_descending).merge(fapar_masked)  # .merge(coherence)
        else:
            sigma_ascending = self._eoconn.load_collection('SENTINEL1_GRD', bands=['VH', 'VV'],properties={"orbitDirection": lambda od: eq(od, "ASCENDING")})
            sigma_ascending = sigma_ascending.sar_backscatter(coefficient="sigma0-ellipsoid",local_incidence_angle=True)
            sigma_descending = self._eoconn.load_collection('SENTINEL1_GRD', bands=['VH', 'VV'], properties={"orbitDirection": lambda od: eq(od, "DESCENDING")})
            sigma_descending = sigma_descending.sar_backscatter(coefficient="sigma0-ellipsoid", local_incidence_angle=True).resample_cube_spatial(sigma_ascending)

            S2mask = create_mask(self._eoconn, scl_layer_band='SENTINEL2_L2A_SENTINELHUB:SCL')
            S2_bands = self._eoconn.load_collection('SENTINEL2_L2A_SENTINELHUB',bands=["B03", "B04", "B08", "sunAzimuthAngles", "sunZenithAngles",
                                                     "viewAzimuthMean", "viewZenithMean"])
            S2_bands_mask = S2_bands.mask(S2mask)
            S2_bands_mask = S2_bands_mask.resample_cube_spatial(sigma_ascending)
            udf = self.load_udf('UDF_biopar_calculation_shub_3_band.py')
            udf = udf.replace('$BIOPAR', "'{}'".format('FAPAR'))
            fapar_masked = S2_bands_mask.reduce_bands_udf(udf)
            fapar_masked = fapar_masked.add_dimension('bands', label = 'band_0', type = 'bands')
            all_bands = sigma_ascending.merge_cubes(sigma_descending).merge_cubes(fapar_masked)

        return all_bands

    @classmethod
    def load_geometry(cls, gjson_path):
        # LOAD THE FIELDS FOR WHICH THE TIMESERIES
        # SHOULD BE EXTRACTED FOR THE CROP CALENDARS
        with open(gjson_path) as f:
            gj = geojson.load(f)
        # ### Buffer the fields 10 m inwards before requesting the TS from OpenEO
        polygons_inw_buffered, poly_too_small_buffer = prepare_geometry(gj)
        # #TODO this is a bit confusing: I woud expect to continue with polygons_inw_buffered here
        gj = remove_small_poly(gj, poly_too_small_buffer)

        return gj,polygons_inw_buffered

    ##### FUNCTION TO BUILD PROCESS GRAPH NEEDED FOR HARVEST PREDICTIONS
    def generate_cropcalendars_workflow(self, start, end, gjson_path, run_local = False):

            #gj = self.load_geometry(gjson_path) #, polygons_inw_buffered
            # geo = shapely.geometry.GeometryCollection(
            #     [shapely.geometry.shape(feature).buffer(0) for feature in polygons_inw_buffered])

            # get the datacube containing the time series data
            bands_ts = self.get_bands(self.shub)

            ##### POST PROCESSING TIMESERIES USING A UDF
            timeseries = bands_ts.filter_temporal(start,end).polygonal_mean_timeseries(gjson_path)
            if run_local:
                return timeseries

            udf = self.load_udf('crop_calendar_udf.py')

            # Default parameters are ingested in the UDF
            context_to_udf = dict({'window_values': self.window_values, 'thr_detection': self.thr_detection, 'crop_calendar_event': self.crop_calendar_event,
                                   'metrics_crop_event': self.metrics_crop_event, 'VH_VV_range_normalization': self.VH_VV_range_normalization,
                                   'fAPAR_range_normalization': self.fAPAR_range_normalization, 'fAPAR_rescale_Openeo': self.fAPAR_rescale_Openeo,
                                   'index_window_above_thr': self.index_window_above_thr,
                                   'metrics_order': self.metrics_order, 'path_harvest_model': self.path_harvest_model,
                                   'shub': self.shub, 'max_gap_prediction': self.max_gap_prediction, 'gjson': gjson_path})

            crop_calendars_graph = timeseries.process("run_udf",data = timeseries._pg, udf = udf, runtime = 'Python', context = context_to_udf)
            # crop_calendars_df = pd.DataFrame.from_dict(crop_calendars)
            return crop_calendars_graph


    def generate_cropcalendars(self, start, end, gjson_path):
        workflow = self.generate_cropcalendars_workflow(start, end, gjson_path)

        # def fwrite(fname, fcontent):
        #     f = open(fname, 'w')
        #     f.write(str(fcontent))
        #     f.close()
        # fwrite(os.path.join(r'S:\eshape\Pilot 1\results\Harvest_date\UDP', 'cropcalendar_udp_version_2021_03_16.json'), workflow.to_json())

        crop_calendars = workflow.send_job().start_and_wait().get_results()
        return crop_calendars.get_asset('out').load_json()

    def generate_cropcalendars_local(self, start, end, gjson_path):
        timeseries = self.generate_cropcalendars_workflow(start, end, gjson_path, run_local= True)
        timeseries = timeseries.execute()
        with open(r"S:\eshape\Pilot 1\results\Harvest_date\Code_testing\Field_BE\Field_BE_TS_20190101_20190731_orbit_direction2.json", 'w') as json_file:
            json.dump(timeseries, json_file)
        with open(r"S:\eshape\Pilot 1\results\Harvest_date\Code_testing\Field_BE\Field_BE_TS_20190101_20190731_orbit_direction2.json", 'r') as json_file:
            ts = json.load(json_file)

        ts_dict = ts
        df_metrics = timeseries_json_to_pandas(ts_dict)
        df_metrics.index = pd.to_datetime(df_metrics.index)

        from .crop_calendar_local import udf_cropcalendars_local
        crop_calendars_df = udf_cropcalendars_local(ts_dict)






