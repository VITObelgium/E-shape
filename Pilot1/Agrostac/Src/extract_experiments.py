# coding: latin-1
from __future__ import division
# from __builtin__ import True
# import clr
# clr.AddReference("Microsoft.Office.Interop.Excel")
# import Microsoft.Office.Interop.Excel as Excel
# from System.Reflection import Assembly
import os.path, sys
import codecs
import datetime as dt
from Pilot1.Agrostac.Src.agrostac import Trial, Event, ObservedData, ObservedTimeSeries
from copy import deepcopy
from Pilot1.Agrostac.Src.country_bbox import latlon_in_bbox
import pandas as pd
import geopandas as gpd
import numpy as np

__author__ = "Steven B. Hoek"

# Define constants
name = "cropCalendars2018Komotini_Greece"
provider_code = "UNKNOWN"
NA = -2146826246


# Prepare to load management info
# PATH_TO_DLL = os.path.abspath("..\\..\\Utils\\Assemblies")
# assemblyPath = os.path.join(PATH_TO_DLL, "LumenWorks.Framework.IO.dll")
# assembly = Assembly.LoadFile(assemblyPath)
# clr.AddReference(assembly)
# from System.IO import StringReader
# from LumenWorks.Framework.IO.Csv import CsvReader

# Now load management info
# f = open(os.path.join("..\\data", "table2.csv"))
# #reader = CsvReader(StringReader(f.read()), True)
# mgmt_info = []
# headers = reader.GetFieldHeaders()
# colCount = reader.FieldCount
# while reader.ReadNextRecord():
#     mydict = {}
#     for i in range(colCount):
#         fldname = headers[i]
#         mydict[fldname] = reader[i]
#     mgmt_info.append(mydict)

def get_date_tuple(value):
    if ("-" in str(value)):
        # Assume yyyy-mm-dd format
        return str(value).split("-")
    else:
        # Assume a figure
        d = dt.date(1899, 12, 30) + dt.timedelta(days=int(value))
        return (d.year, d.month, d.day)


def get_mgmt_info(acode, anumber):
    result = []
    try:
        result = filter(lambda v: (v["Code"] == acode) and (v["Treatment"] == str(anumber)), mgmt_info)
    finally:
        return result


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


def main():
    # Initialise variables indicating paths etc.
    curdir = r'S:\eshape\Pilot 1\data\Parcels_greece'
    datapath = os.path.join(curdir)
    datapath = os.path.normpath(datapath)
    outdatapath = os.path.normpath(r'S:\eshape\Pilot 1\Agrostac\SIF_files')
    fn0 = os.path.realpath(os.path.join(datapath, "cropCalendars2018Komotini_cropcalendars.xlsx"))
    fn1 = os.path.join(outdatapath, "cropCalendars2018Komotini_cropcalendars.sif")
    fout = None

    # Initialise Excel
    # global excel
    # excel = Excel.ApplicationClass()
    # excel.Visible = True
    # excel.DisplayAlerts = False
    # wb = None

    # TODO: The script below is very sensitive to changes in worksheets 1, 6 and 9. The rows on these worksheets must correspond
    # in terms of experiment name or "exname" and treatment number. The latter is indicated by % or #. The script should improved,
    # so that any inconsistency is detected immediately or is not even dependent on the mentioned correspondence anymore.
    try:
        # Open the Excel file
        if (not os.path.exists(fn0)) or (not os.path.isfile(fn0)):
            raise IOError("File %s does not exist" % fn0)
        else:
            print("Opening file " + fn0)
        try:
            wb = pd.DataFrame(gpd.read_file(fn0))  # excel.Workbooks.Open(fn0)
        except:
            wb = pd.read_excel(fn0)
        # ws1 = wb.Worksheets[1] # metadata
        # ws6 = wb.Worksheets[6] # management events
        # ws9 = wb.Worksheets[9] # observed data summary

        # # Read worksheet 10 into a structure which we can query
        # ws10 = wb.Worksheets[10] # observed data time series
        # observed_timeseries = ObservedTimeSeries()
        # for i in range(2, 1136):
        #     try:
        #         rng = ws10.Range("A" + str(i), "R" + str(i))
        #         od = ObservedData(str(rng.Value2[0, 0]), int(rng.Value2[0, 1]))
        #         datetpl = get_date_tuple(rng.Value2[0, 2])
        #         od.set_date(dt.date(int(datetpl[0]), int(datetpl[1]), int(datetpl[2])))
        #         tmp = rng.Value2[0, 4]
        #         if tmp != NA: od.set_storage_organ_fresh_weight(float(tmp)*1000) # UYFAD
        #         tmp = rng.Value2[0, 5]
        #         if tmp != NA: od.set_storage_organ_dry_weight(float(tmp)) # UWAD
        #         tmp = rng.Value2[0, 9]
        #         if tmp != NA: od.set_lai(float(tmp)) # LAID
        #         tmp = rng.Value2[0, 11]
        #         if tmp != NA: od.set_above_ground_biomass_dry_weight(float(tmp)) # CWAD
        #         observed_timeseries.append(od)
        #
        #         # Extra checks
        #         fw = od.get_storage_organ_fresh_weight()
        #         dw = od.get_storage_organ_dry_weight()
        #         if (fw != -1) and (dw != -1) and (fw < dw):
        #             errmsg = "Fresh weight is less than dry weight for experiment with code %s and treatment no. %s"
        #             raise Warning(errmsg % (od.get_code(), od.get_treatment_number()))
        #     except Exception as e:
        #         print ("Error whilst parsing line %s: %s" % (i, str(e)))
        #
        # # An extra worksheet was added with information from the article
        # objectives = []
        # ws11 = wb.Worksheets[11] # objective
        # for i in range(2, 45):
        #     rng = ws11.Range("A" + str(i), "B" + str(i))
        #     objectives.append({'code': str(rng.Value2[0, 0]), 'objective': str(rng.Value2[0, 1])})

        # Open the input and prepare the output file
        fout = codecs.open(fn1, mode='w', encoding="utf-8")
        s = "// Deze file moet met Merge geladen worden omdat de tijdseries voor dezelfde locatie over verschillende secties verdeeld zijn"
        fout.write(s + "\n\n")

        # # After the header row, there are 216 rows with data in the first worksheet
        # for i in range(2, 218):
        #     try:
        #         rng = ws1.Range("A" + str(i), "U" + str(i))
        #         treatment_number = int(rng.Value2[0, 0])
        #         exname = str(rng.Value2[0, 3])
        #         treatment = str(rng.Value2[0, 5])
        #         if (" " in treatment): treatment = treatment.replace(" ", "_")
        #         organisation = str(rng.Value2[0, 6])
        #         location = str(rng.Value2[0, 7])
        #         country = str(rng.Value2[0, 20])
        #         tmp = rng.Value2[0, 13]
        #         if tmp != NA: latitude = tmp
        #         tmp = rng.Value2[0, 14]
        #         if tmp != NA: longitude = tmp
        #         tmp = rng.Value2[0, 15]
        #         if tmp != NA: altitude = tmp
        for i in range(wb.shape[0]):
            print('CONVERTED FIELD {} OUT OF {} TO SIF FILE'.format(str(i), str(wb.shape[0])))
            try:
                exname = wb.iloc[i, :]['id']
                try:
                    treatment = str(wb.iloc[i, :]['treatments'])
                except:
                    treatment = "UNKNOWN"
                organisation = 'VITO'
                country = 'Greece'
                if country == 'Greece':
                    shp_fields_Greece = gpd.read_file(
                        r"S:\eshape\Pilot 1\data\Parcels_greece\35TLF_2018_parcel_ids_greece.shp")
                    shp_fields_Greece_id = shp_fields_Greece.loc[shp_fields_Greece.id == exname]
                try:
                    location = wb.iloc[i, :]['community']
                except:
                    location = 'UNKNOWN'

                if not country == 'Greece':
                    longitude = wb.iloc[i, :].geometry.centroid.bounds[0]
                    latitude = wb.iloc[i, :].geometry.centroid.bounds[1]
                else:
                    longitude = shp_fields_Greece_id.geometry.values[0].centroid.bounds[0]
                    latitude = shp_fields_Greece_id.geometry.values[0].centroid.bounds[1]

                # Specify some general properties
                if treatment == 'UNKNOWN':
                    trial = Trial(exname)
                else:
                    trial = Trial(exname + '/' + treatment)

                if not country == "Greece":
                    trial.crop = wb.iloc[i, :]['croptype']
                else:
                    trial.crop = 'cotton'
                try:
                    trial.cultivar = wb.iloc[i, :]['variety']
                except:
                    trial.cultivar = "UNKNOWN"
                trial.name = name + "/"
                trial.country = country
                trial.region = location
                trial.site = str(wb.iloc[i, :]['id'])
                trial.organisation = organisation
                trial.provider = provider_code
                trial.lon = longitude
                trial.lat = latitude
                # trial.alt = altitude
                if not latlon_in_bbox((float(trial.lat), float(trial.lon)), trial.country):
                    raise ValueError("Inconsistent input")

                trial.events = []

                ### add some info on the observed phenology to the sif file
                if not country == "Greece":
                    datepl_planting = wb.iloc[i, :]['planting_d']
                else:
                    datepl_planting = wb.iloc[i, :]['Planting_date']
                if datepl_planting != None and not pd.isnull(datepl_planting):
                    datepl_planting = get_date_tuple(datepl_planting)
                    plantingdate = dt.date(int(datepl_planting[0]), int(datepl_planting[1]),
                                           int(datepl_planting[2][0:2]))
                    trial.events.append(Event(plantingdate, "CROP_DEV_BBCH", "00"))

                datepl_harvest = wb.iloc[i, :]['harvest_da']
                if datepl_harvest != None and not pd.isnull(datepl_harvest):
                    datepl_harvest = get_date_tuple(datepl_harvest)
                    harvestdate = dt.date(int(datepl_harvest[0]), int(datepl_harvest[1]), int(datepl_harvest[2][0:2]))
                    trial.events.append(Event(harvestdate, "CROP_DEV_BBCH", "99"))
                if country == 'Greece':
                    datepl_emergence = wb.iloc[i, :]['Emergence_date']
                    if datepl_emergence != None and not pd.isnull(datepl_emergence):
                        datepl_emergence = get_date_tuple(datepl_emergence)
                        emergencedate = dt.date(int(datepl_emergence[0]), int(datepl_emergence[1]),
                                                int(datepl_emergence[2][0:2]))
                        trial.events.append(Event(emergencedate, 'CROP_DEV_BBCH', '9'))

                # # Add information from the CSV file
                # if len(mgmt_info) == 0: print("No management info found!")
                # mgmt_type = get_mgmt_info(exname, treatment_number)
                # if len(mgmt_type) == 0:
                #     trial.water_mgmt_type = "UNKNOWN"
                #     trial.nutrients_mgmt_type = "UNKNOWN"
                #     trial.nutrients_n_type = "UNKNOWN"
                #     trial.nutrients_p_type = "UNKNOWN"
                #     trial.nutrients_k_type = "UNKNOWN"
                # else:
                #     trial.water_mgmt_type = translate_mgmt_info(mgmt_type[0]["WaterManagement"])
                #     trial.nutrients_mgmt_type = translate_mgmt_info(mgmt_type[0]["NutrientsManagement"])
                #     trial.nutrients_n_type = translate_mgmt_info(mgmt_type[0]["NutrientsNType"])
                #     trial.nutrients_p_type = translate_mgmt_info(mgmt_type[0]["NutrientsPType"])
                #     trial.nutrients_k_type = translate_mgmt_info(mgmt_type[0]["NutrientsKType"])
                #
                #
                # # Check a few things
                # if not latlon_in_bbox((float(trial.lat), float(trial.lon)), trial.country):
                #     raise ValueError("Inconsistent input")
                #
                # # Retrieve some more particulars from worksheet 6
                # rng = ws6.Range("A" + str(i), "U" + str(i))
                # assert str(rng.Value2[0, 0]) == exname, "Not the same experiment on line %s" % i
                # assert int(rng.Value2[0, 1]) == treatment_number, "Not the same treatment on line %s" % i
                # trial.events = []
                # dates = []
                # datetpl = get_date_tuple(rng.Value2[0, 2])
                # startdate = dt.date(int(datetpl[0]), int(datetpl[1]), int(datetpl[2]))
                # dates.append(startdate)
                # trial.events.append(Event(startdate, 'CROP_DEV_BBCH', '00'))
                # if rng.Value2[0, 3] != NA:
                #     datetpl = get_date_tuple(rng.Value2[0, 3])
                #     emergencedate = dt.date(int(datetpl[0]), int(datetpl[1]), int(datetpl[2]))
                #     if emergencedate > startdate:
                #         dates.append(emergencedate)
                #         trial.events.append(Event(emergencedate, 'CROP_DEV_BBCH', '9'))
                #     else: print("Check dates for experiment " + exname)
                #
                # datetpl = get_date_tuple(rng.Value2[0, 4])
                # harvestdate = dt.date(int(datetpl[0]), int(datetpl[1]), int(datetpl[2]))
                # dates.append(harvestdate)
                # trial.events.append(Event(harvestdate, 'CROP_DEV_BBCH', '99'))
                # for mydate in dates:
                #     tmp = rng.Value2[0, 6]
                #     if (tmp != NA):
                #         # trial.events.append(Event(mydate, "CUL_NAME", str(tmp).title()))
                #         trial.cultivar = str(tmp).title()
                #     tmp = rng.Value2[0, 7]
                #     if (tmp != NA):
                #         if (str(tmp).find('industrial') != -1) or (str(tmp).find('processing') != -1):
                #             # trial.events.append(Event(mydate, 'CUL_NOTES', 'For industrial use'))
                #             trial.intended_use = 'For industrial use'
                #         else:
                #             if (str(tmp).find('table') != -1):
                #                 # trial.events.append(Event(mydate, 'CUL_NOTES', 'For table use'))
                #                 trial.intended_use = 'For table use'
                #     tmp = rng.Value2[0, 8]
                #     vv = filter(lambda v: (v.date == mydate) & (v.code == 'PLANT_DENSITY_CNT_M2'), trial.events)
                #     if (tmp != NA) and len(vv) == 0:
                #         trial.events.append(Event(mydate, 'PLANT_DENSITY_CNT_M2', float(rng.Value2[0, 8])))
                #
                # # Retrieve some more particulars from worksheet 9
                # rng = ws9.Range("A" + str(i), "U" + str(i))
                # assert str(rng.Value2[0, 0]) == exname, "Not the same experiment on line %s" % i
                # assert int(rng.Value2[0, 1]) == treatment_number, "Not the same treatment on line %s" % i
                # tmp = rng.Value2[0, 7]
                # if tmp != NA: trial.events.append(Event(harvestdate, 'SO_DWT_KGHA', float(tmp))) # UWAH
                # tmp = rng.Value2[0, 8]
                # if tmp != NA: trial.events.append(Event(harvestdate, 'SO_FWT_KGHA', float(tmp)*1000)) # UYAFH
                #
                # # Retrieve some more particulars from worksheet 10
                # recs = observed_timeseries.filter(exname, treatment_number)
                # for rec in recs:
                #     # Add extra records but make sure no duplicate data are added
                #     if 'lai' in rec:
                #         vv = filter(lambda v: (v.date == rec['date']) & (v.code == 'LAIG'), trial.events)
                #         if len(vv) == 0: trial.events.append(Event(rec['date'], 'LAIG', rec['lai']))
                #     dw = None
                #     if 'storage_organ_dry_weight' in rec:
                #         vv = filter(lambda v: (v.date == rec['date']) & (v.code == 'SO_DWT_KGHA'), trial.events)
                #         if len(vv) == 0: trial.events.append(Event(rec['date'], 'SO_DWT_KGHA', rec['storage_organ_dry_weight']))
                #         dw = rec['storage_organ_dry_weight']
                #     fw = None
                #     if 'storage_organ_fresh_weight' in rec:
                #         vv = filter(lambda v: (v.date == rec['date']) & (v.code == 'SO_FWT_KGHA'), trial.events)
                #         if len(vv) == 0: trial.events.append(Event(rec['date'], 'SO_FWT_KGHA', rec['storage_organ_fresh_weight']))
                #         fw = rec['storage_organ_fresh_weight']
                #
                #     # Nogmaals een controle dat er geen regelpaar wordt weggeschreven met fresh weight < dry weight!
                #     if (dw != None) and (fw != None) and (fw < dw):
                #         errmsg = "Fresh weight is less than dry weight for experiment with code %s and treatment no. %s"
                #         raise Warning(errmsg % (exname, treatment_number))
                #     if 'above_ground_biomass_dry_weight' in rec:
                #         vv = filter(lambda v: (v.date == rec['date']) & (v.code == 'TOPS_DWT_KGHA'), trial.events)
                #         if len(vv) == 0: trial.events.append(Event(rec['date'], 'TOPS_DWT_KGHA', rec['above_ground_biomass_dry_weight']))
                #
                # # Finally retrieve the objective
                # recs = filter(lambda x: x['code'] == exname, objectives)
                # if len(recs) > 0:
                #     trial.objective = "'" + recs[0]['objective'] + "'"

                # Now write the trial with all its events
                fout.write(str(trial))
                fout.flush()

                '''
                # Now loop over the data
                for row in reader:
                    # Specify some special particulars
                    trial.events = []
                    trial.context = ""
                    startdate = dt.date(int(row["Year"]), 2, 1) + dt.timedelta(int(row['Onset']))
                    trial.events.append(Event(startdate, 'CROP_DEV_BBCH', '00'))
                    harvest_date = dt.date(int(row["Year"]), 7, 31)

                    # Prepare to write output for 3 additional treatments
                    trial0 = deepcopy(trial)
                    trial1 = deepcopy(trial)
                    trial2 = deepcopy(trial)

                    # Output the data wrt. the control replication
                    trial0.id = "Nofert"
                    trial0.nutrients_mgmt_type = "NOT" 
                    trial0.nutrients_n_type = "NOT"
                    trial0.nutrients_p_type = "NOT"
                    trial0.nutrients_k_type = "NOT"
                    trial0.events.append(Event(harvest_date, 'CROP_DEV_BBCH', '99',  "YM"))
                    evt = Event(harvest_date, 'SO_DWT_KGHA', 1000 * float(row['GR'+trial0.id]), "YM")
                    trial0.events.append(evt)          
                    if len(trial0.events) > 0:
                        fout.write(str(trial0))
                '''

                # print "Data for %s trials were written to file %s" % (i, fout.name)
            except Exception as e:
                print("Unable to write data to file for trial found on line %s" % i)
                print(str(e))
    finally:
        if fout != None:
            fout.flush()
            fout.close()


if __name__ == "__main__":
    main()
