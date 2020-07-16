import openeo
from openeo.rest.conversions import timeseries_json_to_pandas
import pandas as pd
from openeo.rest.job import RESTJob
from Crop_calendars_harvest.Main_functions import OpenEO_extraction_cropSAR
import zipfile
from Pilot1.Emergence_date.Scripts.Extract_ID_emergence_data_available import check_crop_calendar_data_availability
import os
import geopandas as gpd
connection = openeo.connect("http://openeo-dev.vgt.vito.be/openeo/0.4.0") #"http://openeo.vgt.vito.be/openeo/0.4.0"
connection.authenticate_basic("bontek","bontek123")
crop_calendar_param = r'planting_d'
dict_meta_files = {'WIG_2019': r"S:\eshape\Pilot 1\data\WIG_data\2019_WIG_planting_harvest_dates_overview_reduc.xlsx"} #'WIG_2018': r"S:\eshape\Pilot 1\data\WIG_data\2018_WIG_planting_harvest_dates_overview_reduc.xlsx",  'Flax_2018' : r"S:\eshape\Pilot 1\data\Flax_fields\vlas_2018_wgs_all_overview.xlsx"
dict_shp_files = {'WIG_2019': r"/data/users/Public/bontek/e_shape/data/2019_WIG_fields_planting_dates_75_100_perc.shp"} #'WIG_2018': r"/data/users/Public/bontek/e_shape/data/2018_WIG_planting_harvest_dates.shp",'Flax_2018': r"/data/users/Public/bontek/e_shape/data/vlas_2018_wgs_all.shp"

Basefolder = r'S:\eshape\Pilot 1\results\Planting_date\S1_S2_data\CropSAR'
ids_data_extraction = check_crop_calendar_data_availability(dict_meta_files, crop_calendar_param)
dict_cropsar = {}
for key in ids_data_extraction:
    shp_dir = dict_shp_files.get(key)
    shp = gpd.read_file(shp_dir)
    shp = shp.loc[shp.id.isin(ids_data_extraction.get(key))]
    outdir = os.path.join(Basefolder,'{}.zip'.format(os.path.split(shp_dir)[-1].split('.')[0]))
    year_interest = int(key.split('_')[-1])
    start_date = '{}-01-01'.format(str(year_interest))
    end_date = '{}-12-31'.format(str(year_interest))
    request = OpenEO_extraction_cropSAR(start_date, end_date, outdir, shp_dir, connection)
    try:
        job = connection.create_job(request["process_graph"], additional= { "driver-memory": "25g","driver-cores": "6","driver-memoryOverhead": "6g","executor-memory": "2g","executor-memoryOverhead": "1500m","executor-cores": "2"})
        RESTJob.run_synchronous(job, outdir)
    except:
       job = connection.create_job(request)
       job.start_job()


    with zipfile.ZipFile(outdir, 'r') as zip_ref:
        zip_ref.extractall('.') # this saves the outcome to the WDIR!!!
    dict_cropsar.update({key: zip_ref})

