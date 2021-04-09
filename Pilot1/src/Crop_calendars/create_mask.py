import numpy as np
import scipy.signal


def create_mask(session, scl_layer_band="TERRASCOPE_S2_TOC_V2:SCENECLASSIFICATION_20M"):
    if scl_layer_band==None:
        scl_layer_band = "TERRASCOPE_S2_TOC_V2:SCENECLASSIFICATION_20M"
    layer_band = scl_layer_band.split(':')
    s2_sceneclassification = session.load_collection(layer_band[0], bands=[layer_band[1]])

    classification = s2_sceneclassification.band(layer_band[1])

    def makekernel(iwindowsize):
        kernel_vect = scipy.signal.windows.gaussian(iwindowsize, std=iwindowsize / 6.0, sym=True)
        kernel = np.outer(kernel_vect, kernel_vect)
        kernel = kernel / kernel.sum()
        return kernel

    #in openEO, 1 means mask (remove pixel) 0 means keep pixel

    #keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 2) | (classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17))
    #remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    first_mask = first_mask > 0.057

    #remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10) | (classification == 11)
    second_mask = second_mask.apply_kernel(makekernel(201))
    second_mask = second_mask > 0.025

    return first_mask | second_mask

