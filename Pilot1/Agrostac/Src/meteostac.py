from __future__ import division
from datetime import datetime, date

__author__ = "Steven B. Hoek"

class WeatherStation():
    __name = 'Unknown'
    __lon = -999
    __lat = -999
    __alt = -999
    __provider = ""
    __comments = ""
    __nodata_value = -99
    __observations = None
    
    def __init__(self, name):
        self.__observations = []
        self.__name = name
        
    def __str__(self):
        result = ''
        try:
            # Write header lines
            result = u":METEO\n\n" 
            result += "Name         = '" + self.name + "'\n"
            result += "WMOCode      = 0" + "\n"
            result += "LongitudeDD  = " + str(self.longitude) + "\n"
            result += "LatitudeDD   = " + str(self.latitude) + "\n"
            result += "AltitudeM    = " + str(self.altitude) + "\n"
            result += "ProviderCode = '" + self.provider + "'\n\n"
            if self.__comments != "":
                result += "Comments     = '" + self.__comments + "'\n\n"
            result += "TimeStampCode,TM_AV_C,TM_MN_C,TM_MX_C,Q_CU_MJM2,PR_CU_MM,WS_AV_MS,RH_AV_PRC\n"
            
            # Write actual data lines    
            self.__observations.sort(key=lambda x: x.date)
            prevdate = datetime(1900, 1, 1, 0, 0)
            for obs in self.__observations:
                if obs.date < prevdate:
                    raise Exception("Found date is not later than previous one!")
                prevdate = obs.date 
                result += str(obs) + "\n"
        except Exception as e:
            print(e)
        finally:
            return result
        
    def append(self, amos):
        if not isinstance(amos, MinimalObservationSet): 
            raise Exception("Received object not of expected type!")
        else:
            amos.nodata_value = self.__nodata_value
            self.__observations.append(amos)
            
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        self.__name = name
        
    @property
    def longitude(self):
        return self.__lon

    @longitude.setter
    def longitude(self, longitude):
        self.__lon = longitude
        
    @property
    def latitude(self):
        return self.__lat

    @latitude.setter
    def latitude(self, latitude):
        self.__lat = latitude
        
    @property
    def altitude(self):
        return self.__alt

    @altitude.setter
    def altitude(self, altitude):
        self.__alt = altitude
    
    @property 
    def provider(self): 
        return self.__provider 

    @provider.setter
    def provider(self, astr):
        self.provider = astr
        
    @property
    def nodata_value(self):
        return self.__nodata_value
    
    @nodata_value.setter
    def nodata_value(self, avalue):
        self.__nodata_value = avalue
        
class MinimalObservationSet():
    # a limited set of observations obtained on the same day
    __date = datetime
    __srad = 0.0
    __tmax = -99.0
    __tmin = -99.0
    __rain = 0.0
    __nodata_value = -99
    
    def __init__(self, adate, data):
        if len(data) == 0: raise Exception("Empty data sequence received!")
        if isinstance(adate, date) or isinstance(adate, datetime):
            self.__date = adate
        else:
            raise Exception("Invalid date received!")
        self.__srad = float(data[0])
        self.__tmax = float(data[1])
        self.__tmin = float(data[2])
        self.__rain = float(data[3])
    
    def fmtval(self, value):
        # Return a string which represents the value best
        if value == self.__nodata_value: result = '-'
        else: result = '%0.1f' % value
        return result
    
    def __str__(self):
        dt = self.__date
        result = 'y' + str(dt.year) + 'm' + str(dt.month).zfill(2) + 'd' + str(dt.day).zfill(2) 
        result += ',' + self.fmtval((self.tmin + self.tmax)/2) + ',' + self.fmtval(self.tmin) + ','
        result += self.fmtval(self.tmax) + ',' + self.fmtval(self.srad) + ',' 
        result += self.fmtval(self.__rain) + ',-,-'
        return result
    
    @property 
    def date(self):
        return self.__date 
    
    @property
    def srad(self):
        return self.__srad
    
    @property
    def tmax(self):
        return self.__tmax
    
    @property
    def tmin(self):
        return self.__tmin
    
    @property
    def rain(self):
        return self.__rain
    