from pathlib import Path
import pandas as pd
import openeo
import shapely
from openeo import Job
from openeo.rest.conversions import timeseries_json_to_pandas

import os
import statistics
import collections
from Crop_calendars.Terrascope_catalogue_retrieval import OpenSearch

from shapely.geometry.polygon import Polygon

from Crop_calendars.create_mask import create_mask
from Crop_calendars.prepare_geometry import prepare_geometry,remove_small_poly

import geojson
import uuid
import json
from openeo.rest.datacube import DataCube

# General approach:
#
# first merge all required inputs into a single multiband raster datacube
# compute timeseries for one or more fields, for all bands (one step)
# do postprocessing of the timeseries:
#   compute cropsar based on cleaned fapar + sigma0
#   combine cropsar + coherence to determine cropcalendar
#   return cropcalendar output in your own json format

class Cropcalendars():
    def __init__(self, fAPAR_rescale_Openeo, coherence_rescale_Openeo, path_harvest_model,VH_VV_range_normalization, fAPAR_range_normalization, metrics_order, connection = None):
        # crop calendar independant variables
        self.fAPAR_rescale_Openeo = fAPAR_rescale_Openeo
        self.coherence_rescale_Openeo = coherence_rescale_Openeo
        self.path_harvest_model = path_harvest_model
        self.VH_VV_range_normalization = VH_VV_range_normalization
        self.fAPAR_range_normalization  = fAPAR_range_normalization
        self.metrics_order = metrics_order

        # openeo connection
        if(connection == None):

            self._eoconn = openeo\
                .connect('https://openeo-dev.vito.be/openeo/1.0.0')\
                .authenticate_basic('eshape', 'eshape123')
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

    def get_bands(self):
        S2mask = create_mask( self._eoconn)
        fapar = self._eoconn.load_collection('TERRASCOPE_S2_FAPAR_V2', bands=['FAPAR_10M'])

        fapar_masked = fapar.mask(S2mask)

        sigma_ascending = self._eoconn.load_collection('S1_GRD_SIGMA0_ASCENDING', bands=["VH", "VV", "angle"])
        sigma_descending = self._eoconn.load_collection('S1_GRD_SIGMA0_DESCENDING',
                                                        bands=["VH", "VV", "angle"]).resample_cube_spatial(
            sigma_ascending)

        fapar_masked = fapar_masked.resample_cube_spatial(sigma_ascending)

        all_bands = sigma_ascending.merge(sigma_descending).merge(fapar_masked)  # .merge(coherence)
        return all_bands

    @classmethod
    def load_geometry(cls, gjson_path):
        # LOAD THE FIELDS FOR WHICH THE TIMESERIES
        # SHOULD BE EXTRACTED FOR THE CROP CALENDARS
        with open(gjson_path) as f:
            gj = geojson.load(f)
        ### Buffer the fields 10 m inwards before requesting the TS from OpenEO
        polygons_inw_buffered, poly_too_small_buffer = prepare_geometry(gj)
        #TODO this is a bit confusing: I woud expect to continue with polygons_inw_buffered here
        gj = remove_small_poly(gj, poly_too_small_buffer)

        return gj,polygons_inw_buffered

    def generate_cropcalendars(self, start, end, gjson_path, window_values, thr_detection, crop_calendar_event, metrics_crop_event, index_window_above_thr):
            ##### FUNCTION TO BUILD A DATACUBE IN OPENEO

            def get_angle(geo, start, end):
                scale = 0.0005
                orbit_passes = [r'ASCENDING', r'DESCENDING']
                dict_df_angles_fields = dict()
                for orbit_pass in orbit_passes:
                    angle = self._eoconn.load_collection('S1_GRD_SIGMA0_{}'.format(orbit_pass), bands = ['angle']).band('angle')
                    try:
                        angle_fields = angle.filter_temporal(start,end).polygonal_mean_timeseries(geo).send_job().start_and_wait().get_result().load_json()
                        df_angle_fields = timeseries_json_to_pandas(angle_fields)
                    except Exception:
                        print('RUNNING IN EXECUTE MODE WAS NOT POSSIBLE ... TRY BATCH MODE')
                        angle.polygonal_mean_timeseries(geo).filter_temporal(start, end).execute_batch('angle_{}.json'.format(orbit_pass))
                        with open('angle_{}.json'.format(orbit_pass), 'r') as angle_file:
                            angle_fields_ts = json.load(angle_file)
                            df_angle_fields = timeseries_json_to_pandas(angle_fields_ts)
                            df_angle_fields.index = pd.to_datetime(df_angle_fields.index).date
                            angle_file.close()
                            os.remove(os.path.join(os.getcwd(),'angle_{}.json'.format(orbit_pass)))

                    new_columns = [str(item) + '_angle' for item in list(df_angle_fields.columns.values)]
                    df_angle_fields.rename(columns = dict(zip(list(df_angle_fields.columns.values), new_columns)), inplace= True)
                    df_angle_fields = df_angle_fields*scale
                    dict_df_angles_fields.update({'{}'.format(orbit_pass): df_angle_fields})
                return dict_df_angles_fields


            # def to find the optimal orbit

            def find_optimal_RO_per_pass(dict_orbit_metadata_frequency_info, dict_angle_orbit_pass):
                RO_orbit_counter = collections.Counter(list(dict_orbit_metadata_frequency_info.values()))
                RO_steepest_angle = max(dict_angle_orbit_pass, key=lambda x: dict_angle_orbit_pass[x])
                # see if the orbit with steepest angle has not a lot fewer coverages compared to the orbit with the maximum coverages. In case this orbit has more than 80% less
                # coverage another orbit is selected
                if RO_orbit_counter.get(RO_steepest_angle) < int(max(list(RO_orbit_counter.values())) * 0.80):
                    RO_orbit_selection = statistics.mode(list(dict_orbit_metadata_frequency_info.values()))
                else:
                    RO_orbit_selection = RO_steepest_angle
                list_orbit_passes = sorted(list(
                    (key) for key, value in dict_orbit_metadata_frequency_info.items() if value == RO_orbit_selection))
                dict_metadata_RO_selection = {list_orbit_passes[0].strftime('%Y-%m-%d'): RO_orbit_selection}
                return dict_metadata_RO_selection, RO_orbit_selection
            def Opensearch_OpenEO_RO_selection(angle_fields,gj,orbit_passes, s):
                # get some info on the RO intersecting the fields by using the Opensearch for filtering data in Terrascope
                dict_descending_orbits_field, dict_ascending_orbits_field = self._open_search.OpenSearch_metadata_retrieval(start, end, gj.features[s])
                for orbit_pass in orbit_passes:
                    if orbit_pass == 'ASCENDING':
                        df_RO_pass = pd.DataFrame(data = dict_ascending_orbits_field.values(), columns = (['RO']), index = dict_ascending_orbits_field.keys())
                    else:
                        df_RO_pass = pd.DataFrame(data = dict_descending_orbits_field.values(), columns = (['RO']), index = dict_descending_orbits_field.keys())

                    df_RO_pass.index = pd.to_datetime(df_RO_pass.index)
                    df_RO_pass = df_RO_pass.tz_localize(None)
                    df_angle_pass = angle_fields['{}'.format(orbit_pass)].iloc[:,s]
                    df_angle_pass.index = pd.to_datetime(df_angle_pass.index)
                    df_angle_pass = df_angle_pass.tz_localize(None)
                    df_pass_combine  = df_RO_pass.merge(df_angle_pass, left_index= True, right_index= True, how = 'inner') # join the RO orbit and angle dataframe based on their index date
                    dict_angle_pass = df_pass_combine.set_index('RO').T.reset_index(drop=True).to_dict(orient='records')[0]
                    columns_df = list(df_pass_combine.columns.values)
                    columns_df = [item for item in columns_df if not 'angle' in item]
                    dict_metadata_pass = df_pass_combine[columns_df].to_dict()[columns_df[0]]
                    if orbit_pass == 'ASCENDING':
                        dict_metadata_ascending_RO_selection, RO_ascending_selection = find_optimal_RO_per_pass(
                            dict_metadata_pass, dict_angle_pass)

                    else:
                        dict_metadata_descending_RO_selection, RO_descending_selection = find_optimal_RO_per_pass(
                            dict_metadata_pass, dict_angle_pass)

                return dict_metadata_ascending_RO_selection, dict_metadata_descending_RO_selection

            ###############################################################
            ###################### MAIN SCRIPT ############################
            ###############################################################

            gj, polygons_inw_buffered = self.load_geometry(gjson_path)
            geo = shapely.geometry.GeometryCollection(
                [shapely.geometry.shape(feature).buffer(0) for feature in polygons_inw_buffered])

            # get some info on the indicence angle covering the fields
            angle_fields = get_angle(geo, start, end)
            orbit_passes = ['ASCENDING', 'DESCENDING']

            # Find the most suitable ascending/descending orbits based
            # on its availability and incidence angle
            # define an unique id per field that will be needed
            # to estimate the crop calendars properly for each field
            unique_ids_fields = []
            dict_ascending_orbits_field = dict()
            dict_descending_orbits_field = dict()
            for s in range(len(gj.features)):
                gj.features[s].properties['id'] = str(uuid.uuid1())
                unique_ids_fields.extend([gj.features[s].properties['id']])
                RO_ascending_selection, RO_descending_selection = Opensearch_OpenEO_RO_selection(angle_fields, gj, orbit_passes, s)
                dict_ascending_orbits_field.update({gj.features[s].properties['id']: RO_ascending_selection})
                dict_descending_orbits_field.update({gj.features[s].properties['id']: RO_descending_selection})

            # get the datacube containing the time series data
            bands_ts = self.get_bands()


            ##### POST PROCESSING TIMESERIES USING A UDF
            timeseries = bands_ts.filter_temporal(start,end).polygonal_mean_timeseries(geo)
            udf = self.load_udf('crop_calendar_udf.py')

            # Default parameters are ingested in the UDF
            context_to_udf = dict({'window_values': window_values, 'thr_detection': thr_detection, 'crop_calendar_event': crop_calendar_event,
                                   'metrics_crop_event': metrics_crop_event, 'VH_VV_range_normalization': self.VH_VV_range_normalization,
                                   'fAPAR_range_normalization': self.fAPAR_range_normalization, 'fAPAR_rescale_Openeo': self.fAPAR_rescale_Openeo,
                                   'coherence_rescale_Openeo': self.coherence_rescale_Openeo,
                                   'RO_ascending_selection_per_field': dict_ascending_orbits_field, 'RO_descending_selection_per_field': dict_descending_orbits_field,
                                   'unique_ids_fields': unique_ids_fields, 'index_window_above_thr': index_window_above_thr,
                                   'metrics_order': self.metrics_order, 'path_harvest_model': self.path_harvest_model})
            job:Job = timeseries.process("run_udf",data = timeseries._pg, udf = udf, runtime = 'Python', context = context_to_udf).send_job()
            crop_calendars = job.start_and_wait().get_result().load_json()
            crop_calendars_df = pd.DataFrame.from_dict(crop_calendars)


            #### FINALLY ASSIGN THE CROP CALENDAR EVENTS AS PROPERTIES TO THE GEOJSON FILE WITH THE FIELDS
            for s in range(len(gj.features)):
                for c in range(crop_calendars_df.shape[1]):  # the amount of crop calendar events which were determined
                    gj.features[s].properties[crop_calendars_df.columns[c]] = \
                    crop_calendars_df.loc[crop_calendars_df.index == unique_ids_fields[s]][crop_calendars_df.columns[c]].values[0]  # the date of the event
            return gj








