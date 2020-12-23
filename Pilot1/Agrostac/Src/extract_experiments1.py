# coding: latin-1
from __future__ import division
from __builtin__ import True
import clr 
clr.AddReference("System.Data")
from System.Data import DataSet
from System.Data.Odbc import OdbcConnection, OdbcDataAdapter, OdbcCommand
from System import DBNull
from agrostac import Trial, Event, ObservedData, ObservedTimeSeries
from country_bbox import latlon_in_bbox
from datetime import datetime, timedelta
import codecs
import os.path, sys

__author__ = "Steven B. Hoek"

name = "doi:10.7910/DVN/V4P6PU"
provider_code = "ODJAR_KASSIE_ET_AL_2018"

def main():
    curdir = os.path.dirname(sys.argv[0])
    datapath = os.path.join(curdir, "../data/15828/")
    datapath = os.path.normpath(datapath)
    fn = os.path.join(datapath, "15828.sif")
    conn = None
    try:
        # Prepare the output file
        fout = codecs.open(fn, mode='w', encoding="utf-8")
        s = "// Deze file moet met Merge geladen worden omdat de tijdseries voor dezelfde locatie over verschillende secties verdeeld zijn"
        fout.write(s + "\n\n")
        
        # Initialise connection to database
        connstr = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};Dbq=' 
        connstr += 'D:\\Userdata\\hoek008\\Agrostac\\data\\15828\\WheatExperiments.accdb;' + 'Uid=Admin;Pwd=';
        conn = OdbcConnection(connstr)
        conn.Open()
        
        # Retrieve the metadata
        sqlstr = "SELECT TreatmentID, Organisation, Location, Country, Exname, TreatmentCode, Longitude, Latitude, Altitude, Objective, Cultivar FROM Metadata"
        adapter = OdbcDataAdapter(sqlstr, conn)
        ds = DataSet()
        adapter.Fill(ds)
        adapter = None
        
        # Now that we have the rows, loop over them
        for row in ds.Tables[0].Rows:
            treatmentid = int(row["TreatmentID"])
            organisation =  str(row["Organisation"])
            location = str(row["Location"])
            country = str(row["Country"])
            exname = str(row["Exname"])
            treatment = str(row["TreatmentCode"])
            longitude = row["Longitude"] 
            latitude = row["Latitude"] 
            altitude =  row["Altitude"] 
            objective = str(row["Objective"])
            cultivar = str(row["Cultivar"])
            
            # Specify some general properties
            trial = Trial(exname + '/' + treatment)
            trial.crop = 'WHB'
            trial.name = name + "/"
            trial.country = country
            pair = location.split(",")
            trial.site = pair[0].strip()
            if len(pair) > 1: trial.region = pair[1].strip()
            else: trial.region = ""
            trial.organisation = organisation
            trial.provider = provider_code 
            trial.lon = longitude
            trial.lat = latitude
            trial.alt = altitude
            trial.objective = "'" + objective + "'"
            trial.cultivar = cultivar
            
            # Check a few things
            if not latlon_in_bbox((float(trial.lat), float(trial.lon)), trial.country):
                raise ValueError("Inconsistent input")
            
            # Retrieve further data
            sqlstr = "SELECT TreatmentID, PDATE, CntPerM2AtPlanting FROM CropManagementEvents WHERE TreatmentID=" + str(treatmentid)
            with OdbcCommand(sqlstr, conn) as command:
                reader = command.ExecuteReader()
                if (reader.HasRows): 
                    reader.Read()
                    tmp = reader.GetDate(1)
                    startdate = datetime(tmp.Year, tmp.Month, tmp.Day)
                    trial.events.append(Event(startdate, 'CROP_DEV_BBCH', '00'))
                    trial.events.append(Event(startdate, 'PLANT_DENSITY_CNT_M2', reader.GetFloat(2)))
                    reader.Close()
                    reader = None
            
            # Load the observations as events        
            sqlstr = "SELECT TreatmentID, date, CWAD, GWAD, LAID FROM Observations WHERE TreatmentID=" + str(treatmentid) 
            with OdbcCommand(sqlstr, conn) as command:
                reader = command.ExecuteReader()
                if (reader.HasRows): 
                    while reader.Read():
                        if not reader.IsDBNull(1):
                            tmp = reader.GetDate(1)
                            if not reader.IsDBNull(2):
                                trial.events.append(Event(datetime(tmp.Year, tmp.Month, tmp.Day), 'TOPS_DWT_KGHA', reader.GetDouble(2)))
                            if not reader.IsDBNull(3):
                                trial.events.append(Event(datetime(tmp.Year, tmp.Month, tmp.Day), 'SO_DWT_KGHA', reader.GetDouble(3)))
                            if not reader.IsDBNull(4):   
                                trial.events.append(Event(datetime(tmp.Year, tmp.Month, tmp.Day), 'LAIG', reader.GetDouble(4)))
                reader.Close()
                reader = None
                
            # Unfortunately, it was not possible to derive the emergence date from the data
            maturitydate = None
            sqlstr = "SELECT TreatmentID, ADAT, MDAT, HWAH, CWAM FROM GrowthSummary WHERE TreatmentID=" + str(treatmentid) 
            with OdbcCommand(sqlstr, conn) as command:
                reader = command.ExecuteReader()
                if (reader.HasRows): 
                    reader.Read()
                    if not reader.IsDBNull(1):
                        tmp = reader.GetDate(1) 
                        anthesisdate = datetime(tmp.Year, tmp.Month, tmp.Day)
                        trial.events.append(Event(anthesisdate, 'CROP_DEV_BBCH', '61'))
                    if not reader.IsDBNull(2):
                        tmp = reader.GetDate(2) 
                        maturitydate = datetime(tmp.Year, tmp.Month, tmp.Day)
                        trial.events.append(Event(maturitydate, 'CROP_DEV_BBCH', '99')) 
                    if not reader.IsDBNull(3) and not maturitydate is None: 
                        vv = filter(lambda v: (v.date == maturitydate.date()) & (v.code == 'SO_DWT_KGHA'), trial.events)
                        if len(vv) == 0: trial.events.append(Event(maturitydate, 'SO_DWT_KGHA', reader.GetDouble(3)))
                    if not reader.IsDBNull(4) and not maturitydate is None:  
                        vv = filter(lambda v: (v.date == maturitydate.date()) & (v.code == 'TOPS_DWT_KGHA'), trial.events)  
                        if len(vv) == 0: trial.events.append(Event(maturitydate, 'TOPS_DWT_KGHA', reader.GetDouble(4))) 
                reader.Close()
                reader = None
                
            sqlstr = "SELECT TreatmentID, WaterManagement, NutrientsManagement, NutrientsNType, NutrientsPType, NutrientsKType "
            sqlstr += "FROM ManagementSummary WHERE TreatmentID=" + str(treatmentid) 
            with OdbcCommand(sqlstr, conn) as command:
                reader = command.ExecuteReader()
                if (reader.HasRows): 
                    reader.Read()
                    trial.water_mgmt_type = translate_mgmt_info(reader.GetString(1))
                    trial.nutrients_mgmt_type = translate_mgmt_info(reader.GetString(2))
                    trial.nutrients_n_type = translate_mgmt_info(reader.GetString(3))
                    trial.nutrients_p_type = translate_mgmt_info(reader.GetString(4))
                    trial.nutrients_k_type = translate_mgmt_info(reader.GetString(5))
                reader.Close()
                reader = None
                        
            # Now write the trial with all its events                
            fout.write(str(trial))
            fout.flush()
            
    except Exception as e:
        print(e)
    finally:
        if not conn is None:
            conn.Close()
        if not fout is None: 
            fout.flush()
            fout.close()

def translate_mgmt_info(aletter):
    result = "UNKNOWN"
    try:
        options = {"N": "NOT",
                   "S": "SUBOPTIMAL",
                   "E": "EXT_SERV_RECOMM",
                   "O": "OPTIMAL",
                   "X": "EXCESSIVE",
                   "U": "UNKNOWN"
                  }
        result = options[aletter]
    finally:
        return result

if __name__ == "__main__":
    main()