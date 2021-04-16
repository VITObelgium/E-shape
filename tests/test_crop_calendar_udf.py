'''
Created on Apr 11, 2021

@author: banyait
'''
import unittest
import geojson
from Crop_calendars.crop_calendar_udf import udf_cropcalendars
from openeo_udf.api.udf_data import UdfData
from openeo_udf.api.structured_data import StructuredData
import json
import tempfile
from pathlib import Path
import pandas
import os
import sys
sys.path.append(os.path.realpath(__file__))
class TestCropCalendarUdf(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
        # load inputs
        with open('resources/WIG_harvest_detection_fields.geojson') as f:
            self.geoms=geojson.load(f)
        with open(os.path.join(os.getcwd(),'resources/WIG_fields_TS_SHUB_20190101_20191231.json')) as f:
            self.ts_shub=geojson.load(f)
            
        # set context
        self.context={
            'window_values': 5, 
            'thr_detection': 0.75, 
            'crop_calendar_event': 'Harvest', 
            'metrics_crop_event': ['cropSAR', 'VH_VV_{}'], 
            'VH_VV_range_normalization': [-13, -3.5], 
            'fAPAR_range_normalization': [0, 1], 
            'fAPAR_rescale_Openeo': 0.005, 
            'index_window_above_thr': 2, 
            'metrics_order': ['sigma_ascending_VH', 'sigma_ascending_VV', 'sigma_ascending_angle', 'sigma_descending_VH', 'sigma_descending_VV', 'sigma_descending_angle', 'fAPAR'], 
            'path_harvest_model': 'resources/model_update1.0_iteration24.h5',
            'shub': True,
            'max_gap_prediction': 24, 
            'gjson': self.geoms
        }
        
        # boot pypspark
        # self.sc=pyspark.SparkContext.getOrCreate()
        
        # set temp dir
        self.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(self):
        self.tmp.cleanup()


    def processTS(self,tsa):
        return [pandas.DataFrame.from_dict(its,orient='index').sort_index() for its in tsa.values() ]

    def compareSignals(self, df0, df1, max_oor_percent, tolerance):
        # raw input
        self.assertTrue(len(df0.index)>0)
        self.assertEqual(len(set(df0.index)-set(df1.index)),0)
        # replace nodata with some finite but way out of range
        df0f=df0.fillna(1000000.).sort_index()
        df1f=df1.fillna(1000000.).sort_index()
        diffcheck=(df0f-df1f)<tolerance
        oor=1.-diffcheck.value_counts(True)[True]
        print("    OUT OF RANGE PERCENT: "+str(oor*100.))
        #print("    NAN PERCENT:          ", 1.-(df0f<1000000.).value_counts(True)[True], 1.-(df1f<1000000.).value_counts(True)[True])
        self.assertTrue(oor<max_oor_percent/100.)

    def test_01_CropCalendarCalculation(self):

        # build udf data
        ud=UdfData()
        ctx=self.context.copy()
        ctx['shub']=True
        ctx['save_debug_path']=self.tmp.name
        ud.user_context=ctx
        ud.set_structured_data_list([StructuredData("Dictionary data",self.ts_shub,"dict")])

        # run udf
        result=udf_cropcalendars(ud)
        res=result.get_structured_data_list()[0].data

        with open(Path(str(self.tmp.name),"WIG_fields_harvest_prediction_20190101_20191231.json"),"w") as f:
            json.dump(res, f)
        
        
    def test_02_CropsarCurveIsSame(self):

        with open("resources/WIG_fields_TS_20190101_20191231_cropsar_input_model.json") as f:
            ref= self.processTS(json.load(f))

        with open(Path(str(self.tmp.name),"WIG_fields_TS_20190101_20191231_cropsar_input_model.json")) as f:
            res=self.processTS(json.load(f))

        self.assertEqual(len(ref),len(res))
        
        for i in range(len(ref)):
            self.compareSignals(ref[i][0], res[i][0], 1., 1.e-5)


    def test_03_InputToModelIsSame(self):

        with open("resources/WIG_fields_TS_20190101_20191231_input_df_model.json") as f:
            ref= self.processTS(json.load(f))

        with open(Path(str(self.tmp.name),"WIG_fields_TS_20190101_20191231_input_df_model.json")) as f:
            res=self.processTS(json.load(f))

        self.assertEqual(len(ref),len(res))
        
        for i in range(len(ref)):
            self.compareSignals(ref[i][0], res[i][0], 1., 1.e-5)


    def test_04_ResultIsSame(self):
        
        with open(Path(str(self.tmp.name),"WIG_fields_harvest_prediction_20190101_20191231.json")) as f:
            res=json.load(f)
        with open("resources/WIG_fields_harvest_prediction_20190101_20191231.json") as f:
            ref=json.load(f)
        
        self.assertEqual(len(res['features']), len(ref['features']))
        
        for i in range(len(ref['features'])):
            vres=res['features'][i]['properties']['Harvest_date']
            vref=ref['features'][i]['properties']['Harvest_date']
            vres= vres if isinstance(vres, str) else "EMPTY"
            vref= vref if isinstance(vref, str) else "EMPTY"
            self.assertEqual(vres,vref)


if __name__ == '__main__':
    unittest.main()