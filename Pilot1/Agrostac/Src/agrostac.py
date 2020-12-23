import datetime as dt

__author__ = "Steven B. Hoek"

class Trial():
    id = 0
    country = ""
    region = ""
    site = ""
    name = ""
    organisation = ""
    lon = -999
    lat = -999
    alt = -999
    crop = ""
    cultivar = ""
    intended_use = ""
    planting_density = 0.0
    provider = ""
    irrigation = False
    institution = ""
    comments = ""
    context = ""
    objective = ""
    events = None
    harvested_yield_dwt = 0.0 # UWAH
    harvested_yield_fwt = 0.0 # UYAFH
    water_mgmt_type = "UNKNOWN"
    nutrients_mgmt_type = "UNKNOWN"
    nutrients_n_type = "UNKNOWN"
    nutrients_p_type = "UNKNOWN"
    nutrients_k_type = "UNKNOWN"
    pest_disease_mgmt_type = "UNKNOWN"
    
    def __init__(self, trial_id):
        self.events = []
        self.id = trial_id
        
    def __str__(self):
        self.comments = self.country
        if self.region != "": self.comments += ", " + self.region
        if self.site != "": self.comments += ", " + self.site
        if self.organisation != "": self.comments += ", " + self.organisation 
        if self.context != "": self.comments += " (" + self.context + ")"
        result = u":CROP_CULTIVATION\n\n"
        result += "Name         = '" + self.name + str(self.id) + "'\n"
        result += "LongitudeDD  = " + str(self.lon) + "\n"
        result += "LatitudeDD   = " + str(self.lat) + "\n"
        result += "AltitudeM    = " + str(self.alt) + "\n"
        result += "ProviderCode = '" + self.provider + "'\n"
        if self.comments != ", , , ":
            result += "Comments     = '" + self.comments + "'\n\n"
        result += "FieldManagementType = FIELD_TRIAL\n"
        result += "WaterManagementType = " + self.water_mgmt_type + "\n"
        result += "NutrientsManagementType = " + self.nutrients_mgmt_type + "\n"
        result += "NutrientsNType = " + self.nutrients_n_type + "\n"
        result += "NutrientsPType = " + self.nutrients_p_type + "\n"
        result += "NutrientsKType = " + self.nutrients_k_type + "\n"
        result += "PestsDiseasesManagementType = " + self.pest_disease_mgmt_type + "\n\n"

        if self.objective != "": result += "Objective = " + self.objective + "\n\n"
        
        result += "TimeStampTripletTable\n" 
        prevdate = dt.date(1900, 1, 1)
        self.events.sort(key=lambda x: x.date)
        for event in self.events:
            # Make sure the crop and cultivar are given once per triplet set
            if event.date != prevdate:
                # If the crop / cultivar is not given, then they are not added
                event.crop = self.crop
                event.cultivar = self.cultivar
                event.intended_use = self.intended_use
                prevdate = event.date 
            result += str(event)
        result += "\n\n"
        return result

class Event():
    date = None
    crop = ""
    cultivar = ""
    intended_use = ""
    code = ""
    value = -99
    detail = ""

    
    def __init__(self, adate, acode, avalue, detail="YMD"):
        self.code = acode
        if (isinstance(adate, str)) and  ('-' in adate):
            adate = adate.replace('-') 
        self.date = check_date(adate)
        self.value = avalue  
        self.detail = detail
        
    def ensure2digits(self, x):
        result = str(x)
        if len(result) == 1: result = '0' + result
        return result 

    def __str__(self):
        datestr = ('Y' + str(self.date.year) + 
            'M' + self.ensure2digits(self.date.month) + 
            'D' + self.ensure2digits(self.date.day))
        result = ""
        if self.value != -99:
            # It's assumed that storage organs and total biomass at harvest are linked to a month
            if (self.detail == "YM"): 
                if self.crop != "":
                    result += datestr[:-3] + ",CROP_CODE,'" + self.crop + "'\n"
                if self.cultivar != "":
                    result += datestr[:-3] + ",CUL_NAME,'" + self.cultivar + "'\n"
                result += datestr[:-3] + "," + self.code + ",'" + str(self.value) + "'\n"
            else:
                if self.crop != "":
                    result += datestr + ",CROP_CODE,'" + self.crop + "'\n"
                if self.cultivar != "":
                    result += datestr + ",CUL_NAME,'" + self.cultivar + "'\n"
                result += datestr + "," + self.code + ",'" + str(self.value) + "'\n"
        return result

class ObservedData(dict):
    __storage_organ_dry_weight = None
    __storage_organ_fresh_weight = None
    def __init__(self, code, treatment_no):
        self['code'] = code
        self['treatment_number'] = treatment_no
        
    def set_date(self, adate):
        self['date'] = adate
    
    # Make it possible to also add relevant data such as above-ground biomass and LAI
    def set_storage_organ_dry_weight(self, avalue):
        self['storage_organ_dry_weight'] = avalue # UWAD
        
    def set_storage_organ_fresh_weight(self, avalue):
        self['storage_organ_fresh_weight'] = avalue # UYFAD
        
    def set_above_ground_biomass_dry_weight(self, avalue):
        self['above_ground_biomass_dry_weight'] = avalue # CWAD
        
    def set_lai(self, avalue):
        self['lai'] = avalue # LAID
        
    def get_storage_organ_dry_weight(self):
        result = -1.0
        if 'storage_organ_dry_weight' in self:
            result = self['storage_organ_dry_weight']
        return result
        
    def get_storage_organ_fresh_weight(self):
        result = -1.0
        if 'storage_organ_fresh_weight' in self:
            result = self['storage_organ_fresh_weight']
        return result

class ObservedTimeSeries(list):
    # idea is to add ObservedData instances to the list
    # TODO: override append method, so that new additions are checked
    
    def filter(self, exp_code, treatment_no):
        f = lambda v: (v["code"] == exp_code) & (v["treatment_number"] == treatment_no)
        result = filter(f, self)
        return result
      
def check_date(indate):
    """Check representations of date and try to force into a datetime.date

    The following formats are supported:

    1. a date object
    2. a datetime object
    3. a string of the format YYYYMMDD
    4. a string of the format YYYYDDD

    Formats 2-4 are all converted into a date object internally.
    """
    if isinstance(indate, dt.datetime):
        return indate.date()
    elif isinstance(indate, dt.date):
        return indate
    elif isinstance(indate, str):
        skey = indate.strip()
        l = len(skey)
        if l==8:
            # assume YYYYMMDD
            dkey = dt.datetime.strptime(skey,"%Y%m%d")
            return dkey.date()
        elif l==7:
            # assume YYYYDDD
            dkey = dt.datetime.strptime(skey,"%Y%j")
            return dkey.date()
        else:
            msg = "Input value not recognized as date: %s"
            raise KeyError(msg % indate)
    else:
        msg = "Input value not recognized as date: %s"
        raise KeyError(msg % indate)
        