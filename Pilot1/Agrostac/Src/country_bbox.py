import shapefile
import os.path
import sys
import time
from difflib import SequenceMatcher

__author__ = "Steven B. Hoek"

def in_bbox(xy, bbox):
    minx = bbox[0]
    maxx = bbox[2]
    miny = bbox[1]
    maxy = bbox[3]
    A = (xy[0] > minx) and (xy[0] < maxx)
    B = (xy[1] > miny) and (xy[1] < maxy)
    return (A and B)  

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    # Time the following
    t1 = time.time()
    t2 = t1
    
    # Check whether given latlon is likely to be in given country
    lon = 124.82
    lat = 8.643
    country = "Philippines"
    try:
        latlon_in_bbox((lat, lon), country)
        t2 = time.time()
    finally:
        print ("Time elapsed: " + str(t2 - t1) )
    
def latlon_in_bbox(latlon, country):
    # Trace the shape file
    result = False
    curdir = os.path.dirname(sys.argv[0])
    datapath = os.path.normpath(os.path.join(curdir, "../geodata"))
    
    try:
        # Test input argument latlon
        if not (isinstance(latlon[0], float) or isinstance(latlon[1], float)):
            raise ValueError("Argument latlon contains a string where a float is expected")
        
        # Open shapefile with country borders of the world
        fn = os.path.join(datapath, "TM_WORLD_BORDERS-0.3.shp")
        if not os.path.exists(fn):
            raise IOError("File not found: " + fn)
        sf = shapefile.Reader(fn)
        shapes = sf.shapes()
    
        # Loop over the shapes
        found = False
        for i in range(len(shapes)): 
            myobj = sf.shapeRecord(i)
            if str(myobj.record[4]) == country:
                found = True
                myshape = myobj.shape
                bbox = myshape.bbox
                xy = latlon[::-1] # reversed
                if in_bbox(xy, bbox):
                    print ("Given point may indeed be located in country " + country)
                    result = True
                    break 
                else:
                    print ("Given point is definitely not located within country " + country + "!")
            
        if not found:
            # Suggest an alternative name for the country, so that there's a match next time
            altname = ""
            maxscore = 0
            for i in range(len(shapes)):
                myobj = sf.shapeRecord(i)
                name = str(myobj.record[4])
                if country in name:
                    score = 0.99
                else:
                    score = similar(country, name)
                if (score > maxscore):
                    altname = name
                    maxscore = score
            print ('Country not found by name %s. Try: %s' % (country, altname))
            
        # Arrange output
        return result

    except Exception as e:
        print (e)

    finally:
        sf = None 
    
if __name__ == "__main__":
    main() 

