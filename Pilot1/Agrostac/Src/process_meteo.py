import os.path
import sys
from datetime import datetime, date, timedelta
from meteostac import WeatherStation, MinimalObservationSet

__author__ = "Steven B. Hoek"

# We assume that the METEO definition file contains unique variables
def main():
    f = None
    variables = []
    try:
        # Open the file with variable definitions and start reading at line 9
        curdir = os.path.dirname(sys.argv[0])
        datapath = os.path.join(curdir, "../data/15828/")
        datapath = os.path.normpath(datapath)
        fn = os.path.join(datapath, "METEO_v1.sif")
        f = open(fn, 'r') 
        for i in range(9): line = f.readline()
        while (line.strip() != ""):
            parts = line.split(",")
            mydict = {"code":parts[0].strip(), "description":parts[1].strip("' "), "unit":parts[2].strip("' "), "dtype":parts[3].strip()}
            variables.append(mydict)
            line = f.readline()
        for i in range(2): line = f.readline()
        while (line != ""):
            # We assume that there are 2 consecutive lines, 1 for the minimum and 1 for the maxmum
            parts = line.split(",")
            varcode = parts[0].strip()
            vv = filter(lambda v: v["code"] == varcode, variables)
            if len(vv) > 0:
                mydict = vv[0]
                mydict["minimum"] = float(parts[1].strip())
                line = f.readline()
                parts = line.split(",")
                if parts[0].strip() != varcode:
                    raise Exception("Unexpected variable name found: " + parts[0].strip())
                mydict["maximum"] = float(parts[1].strip())
            else: line = f.readline()
            line = f.readline()
        f.close()
        f = None                

        # In the files with meteo data which we're going to process, different abbreviations are used to indicate the variables
        varmatches = [('SRAD', 'Q_CU_MJM2'), ('TMAX', 'TM_MX_C'), ('TMIN', 'TM_MN_C'), ('RAIN', 'PR_CU_MM')]
        
        # Loop over the various stations
        stations = ['AUST', 'BOUW', 'EEST8203', 'MARA9206', 'MARB9206', 'MEXI8812', 'NEWZLAND', 'PAGV8203']
        provider_code = "ODJAR_KASSIE_ET_AL_2018"
        for s in stations:
            # Open the file with meteo data for this station ad prepare an output file too
            print("About to process file " + s + '.WTH')
            fn = os.path.join(datapath, 'Weather data', s + '.WTH')
            f = open(fn, 'r')
            fout = open(os.path.join(datapath, s + '.sif'), 'w')
            
            # Retrieve info from the header lines
            for i in range(4): line = f.readline()
            parts = filter(None, line.split())
            station = WeatherStation(parts[0].strip())
            station.latitude = float(parts[1].strip())
            station.longitude = float(parts[2].strip())
            station.altitude = float(parts[3].strip())
            station.provider = provider_code
            
            # Now loop over the lines with data
            for i in range(2): line = f.readline()
            while (line.strip() != ""):
                parts = filter(None, line.split())
                mydate = convert_date_str(parts[0])
                for i in range(4):
                    # Check each value
                    vv = filter(lambda v: v["code"] == varmatches[i][1], variables)
                    if len(vv) == 0: raise Exception("Variable could not be matched!")
                    vardef = vv[0]
                    value = float(parts[1:5][i])
                    if (value != station.nodata_value) and not check_observation(vardef, value):
                        raise Exception("Invalid value detected for variable %s (%s)" % (varmatches[i][0], mydate))
                amos = MinimalObservationSet(mydate, parts[1:5])
                station.append(amos)

                # Prepare for next loop
                line = f.readline()
                
            # Arrange output
            fout.write(str(station))
            fout.flush()
            station = None
                
            # Prepare for next station
            f.close()
            f = None
            
    finally:
        if not f is None: f.close()

def check_observation(vardef, value):
    return ((value >= vardef["minimum"]) and (value <= vardef["maximum"]))

def convert_date_str(astr):
    yy = astr[0:2]
    doy = int(astr[2:5])
    if int(yy) < 60: yr = int('20' + yy)
    else: yr = int('19' + yy)
    return datetime(yr, 1, 1) + timedelta(doy - 1)

if __name__ == "__main__":
    main()