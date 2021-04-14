import unittest

import pandas
import json
from Crop_calendars.Crop_calendars_openeo_integration import Cropcalendars
import geojson

    
class TestInputs(unittest.TestCase):

    def processTS(self,tsa):
        # remove [[][][]] entries
        tsa=dict(filter(lambda i: len(i[1][0])>0,tsa.items()))
        # split to individual timeseries
        nts=len(next(iter(tsa.items()))[1])
        tss=[
            dict(map(lambda i: (i[0],i[1][its]), tsa.items()))
            for its in range(nts)
        ]
        # convert to dataframes
        dfs=[
            pandas.DataFrame.from_dict(its,orient='index')
            for its in tss
        ]
        return dfs


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

    def test_TimeSeries(self):
        
        # config
        with open("tests/resources/Field_BE.geojson") as f: gj = geojson.load(f)
        start = '2019-01-01'
        end = '2019-07-31'
        
        # query openeo
        cp=Cropcalendars(None, None, None, None, None, None, None, None, None, None, True, None, None)
        cp.fapar_udf_path= 'Pilot1/src/Crop_calendars/UDF_biopar_calculation_shub_3_band.py'
        inppg=cp.generate_cropcalendars_workflow(start, end, gj, run_local=True)
        result=inppg.execute()
        res=self.processTS(result)
        
        # reference
        with open("tests/resources/timeseries_shub_ref.json") as f: 
            ref= self.processTS(json.load(f))
        
        # FOR UPDATING REFERENCE DTA, KEEP IT COMMENTED!
        #with open("tests/resources/timeseries_shub_ref.json","w") as f: 
        #    res= self.processTS(json.load(f))

        # check closeness
        for i in range(len(ref)):
            print("TIMESERIES "+str(i))
            for ivar in range(ref[i].shape[1]):
                print("  VAR "+str(ivar))  
                self.compareSignals(ref[i][ivar], res[i][ivar], 1., 1.e-5)

        print("DONE")
