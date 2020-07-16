"""
https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-body.html#request-body-search-from-size
"""
import pandas as pd
from elasticsearch import Elasticsearch
import os
from shapely.geometry import Polygon
import geopandas as gpd

user = 'fcfa77fc-e266-4e8e-a202-7bbbdff0d329'
es = Elasticsearch(hosts=['es-apps-01.vgt.vito.be', 'es-apps-02.vgt.vito.be', 'es-apps-03.vgt.vito.be'])
query = {
    "query": {
        "bool": {
            "must": [
                {
                    "term": {
                        "applicationId": "databio"
                    }
                },
                {
                    "bool": {
                        "should": [
                            {
                                "term": {
                                    "metadata.owner": user
                                }
                            },
                            {
                                "term": {
                                    "metadata.grantedPermission": user
                                }
                            }
                        ]
                    }
                }
            ],
            "filter": [
                {
                    "range": {
                        "metadata.startDate": {
                            "gte": "2019-01-01",
                            "lte": "2019-12-31"
                        }
                    }
                }
            ]
        }
    }
}

results = es.search(index='databio-fields', body=query, size=10000)['hits']['hits']
print('{} results found'.format(len(results)))

##### extraction of variables per field on the account
df_monitoring_experiment_2019 = []
df_phenology_2019 = []
for p in range(len(results)):
    variety = results[p].get('_source').get('cropFenology').get('variety')
    id = results[p].get('_id')
    fieldname = results[p].get('_source').get('metadata').get('fieldName')
    area = results[p].get('_source').get('metadata').get('area')
    df_phenology_field = pd.DataFrame.from_dict(results[p].get('_source').get('fenologies'))
    try:
        df_phenology_field['id'] = [id]* df_phenology_field.shape[0]
    except:
        df_phenology_field  =pd.DataFrame()
    try:
        planting_date = results[p].get('_source').get('cropFenology').get('plantingDate')
    except:
        planting_data = None
    community = results[p].get('_source').get('metadata').get('community')
    croptype = results[p].get('_source').get('cropFenology').get('cropType')
    try:
        harvest_da = results[p].get('_source').get('harvest')[0].get('date')
    except:
        harvest_da = None
    postal_code = results[p].get('_source').get('metadata').get('postalCode')
    end_year = results[p].get('_source').get('metadata').get('endDate').split('-')[0]

    longitude_list = [results[p].get('_source').get('metadata').get('geometry').get('coordinates')[0][s][0] for s in range(len(results[p].get('_source').get('metadata').get('geometry').get('coordinates')[0]))]
    latitude_list = [results[p].get('_source').get('metadata').get('geometry').get('coordinates')[0][s][1] for s in range(len(results[p].get('_source').get('metadata').get('geometry').get('coordinates')[0]))]
    geometry = Polygon(zip(longitude_list, latitude_list))
    ### make dataframe of  it all:
    df_phenology_field['croptype'] = [croptype] * df_phenology_field.shape[0]
    df_phenology_2019.append(df_phenology_field)
    df_monitoring_experiment_2019.append(pd.concat([pd.DataFrame([variety], index = [str(p)], columns= (['variety'])),
                                              pd.DataFrame([id],index = [str(p)],columns=(['id'])),
                                             pd.DataFrame([fieldname], index= [str(p)], columns=(['fieldame'])),
                                             pd.DataFrame([area], index=[str(p)], columns=(['area'])),
                                             pd.DataFrame([planting_date],index= [str(p)], columns= (['planting_date'])),
                                             pd.DataFrame([community], index= [str(p)], columns= (['community'])),
                                             pd.DataFrame([croptype], index= [str(p)], columns= (['croptype'])),
                                             pd.DataFrame([harvest_da], index= [str(p)], columns= (['harvest_da'])),
                                             pd.DataFrame([postal_code], index= [str(p)], columns= (['postal_code'])),
                                             pd.DataFrame([end_year], index= [str(p)], columns= (['end_year'])), pd.DataFrame([geometry], index= [str(p)], columns= (['geometry']))],axis= 1))

df_phenology_2019 = pd.concat(df_phenology_2019)
df_phenology_2019 = df_phenology_2019.to_csv(r'O:\calval\Crop_extract_cal\Crop_fields_stats\Reference_data\TAP\Development_stages_fields.csv', index = False)
df_monitoring_experiment_2019 = pd.concat(df_monitoring_experiment_2019)
if not os.path.exists(r'S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.csv'):
    print('h')
    #df_monitoring_experiment_2019.to_csv(r'S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.csv', index = False)

#### create shapefile of the fields in the monitoring experiment
gdf_monitoring_experiment_2019 = gpd.GeoDataFrame(df_monitoring_experiment_2019, geometry='geometry')
gdf_monitoring_experiment_2019.crs = "EPSG: 4326"
#gdf_monitoring_experiment_2019.to_file(r"S:\eshape\Pilot 1\data\TAP_monitoring_experiment\2019_TAP_monitoring_experiment.shp", driver = 'ESRI Shapefile')
