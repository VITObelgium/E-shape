**E-shape crop calendars repository** 

In this repository the code used for the cropcalendars is provided. Currently only the automatic harvest detector is already available.

**Requirements and installation** 

° **Python 3.7 or higher environment** (virtual or conda environment) where some packages need to be installed. Consult the '**e_shape_requirements.txt**' file in this folder to install the required packages in your environment. 
  Note: for the installation of openeo you might have to check: https://github.com/Open-EO/openeo-python-client
  
  **Use of code**
 
 The harvest detector code can be initiated with this main script: **Pilot1 -> src -> Crop_calendars -> Main_crop_calendars_openeo_integration.py**.
 In this main file it is possible to predict the harvest date for some field polygons. 
 Note that currently the harvest detector only works for fields in Belgium. Furthermore, the harvest detector will only predict one harvest date for the given time range of interest. 
 
 **Input requirements**
 
 In order to run the harvest detector please be sure that your input file with the fields is a .geojson file with as projection ('WGS84').
 For further specifications on the input requirements, please consult the  Main_crop_calendars_openeo_integration.py script under the section 'USER SPECIFIC PARARMETERS'. The other parameters don't need to change. 