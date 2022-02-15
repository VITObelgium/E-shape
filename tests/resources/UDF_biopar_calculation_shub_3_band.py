import numpy as np
from typing import Dict
from openeo.udf.xarraydatacube import XarrayDataCube
import tensorflow as tf
from biopar.bioparnnw import BioParNNW


biopar_version = '3band'

def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    valid_biopars= ['FAPAR','LAI','FCOVER','CWC','CCC']
    biopar = context.get("biopar", "FAPAR")
    if biopar not in valid_biopars:
        biopar = 'FAPAR'

    ds = cube.get_array()
    ds_date = ds

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

    return XarrayDataCube(xr_biopar)  # xarray.DataArray(as_image,vza.dims,vza.coords)

