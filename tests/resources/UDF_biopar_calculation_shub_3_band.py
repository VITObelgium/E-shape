# import openeo
# import scipy.signal
# import shapely
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenEO Python UDF interface"""

if __name__ == "__main__":
    import doctest

    doctest.testmod()

# from Products.Indices.s2_20m_biopar_processor import S2BioparProcessor
import numpy as np
# from openeo_udf.api.structured_data import StructuredData
from typing import Dict
# import xarray
# from scipy.signal import savgol_filter
from openeo_udf.api.datacube import DataCube
import tensorflow as tf

# from openeo_udf.api.udf_data import UdfData
biopar_version = '3band'
import sys

sys.path.append(r'/data/users/Public/bontek/Nextland/Biopar_test/wheel/biopar-1.2.0-py3-none-any.whl')
from biopar.bioparnnw import BioParNNW


def apply_datacube(cube: DataCube, context: Dict) -> DataCube:
    biopar = $BIOPAR
    ds = cube.get_array()
    ds_date = ds
    # dates = list(ds.coords['t'].values)

    # ds_date = ds.sel(t=date)
    from numpy import cos, radians
    ### LOAD THE DIFFERENT REQUIRED BANDS FOR THE 8-BAND FAPAR
    scaling_bands = 0.0001

    saa = ds_date.sel(bands='sunAzimuthAngles')
    sza = ds_date.sel(bands="sunZenithAngles")
    vaa = ds_date.sel(bands="viewAzimuthMean")
    vza = ds_date.sel(bands="viewZenithMean")

    B03 = ds_date.sel(bands='B03') * scaling_bands
    B04 = ds_date.sel(bands='B04') * scaling_bands
    B8 = ds_date.sel(bands='B08') * scaling_bands


    g1 = cos(radians(vza))
    g2 = cos(radians(sza))
    g3 = cos(radians(saa - vaa))

    #### FLATTEN THE ARRAY ####
    flat = list(map(lambda arr: arr.flatten(),
                    [B03.values, B04.values,B8.values, g1.values, g2.values, g3.values]))
    bands = np.array(flat)

    #### CALCULATE THE BIOPAR BASED ON THE BANDS #####
    image = BioParNNW(version='3band', parameter=biopar, singleConfig = True).run(bands, output_scale=1,
                                                                  output_dtype=tf.dtypes.float32,
                                                                  minmax_flagging=False)  # netcdf algorithm
    as_image = image.reshape((g1.shape))
    ## set nodata to nan
    as_image[np.where(np.isnan(B03))] = np.nan
    xr_biopar = vza.copy()
    xr_biopar.values = as_image

    return DataCube(xr_biopar)  # xarray.DataArray(as_image,vza.dims,vza.coords)

###### OLD CODE #######
# processor = S2BioparProcessor(basepath= r'data', output_dir= r'data', parameter = 'FAPAR')
# variable_array = processor.compute_biopar_S2(ds_date)[::-1]
# mean_time = np.nanmean(variable_array)
# dict_biopar_mean_time.update({date: mean_time})
# udf_data.set_structured_data_list([StructuredData(description="fAPAR_mean json", data=dict_biopar_mean_time, type="dict")])
### how to drop the bands of the xarray
# ds.where(ds.bands == 'RAA_60M', drop=True)




