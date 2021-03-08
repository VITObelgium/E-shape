import numpy as np
import scipy
from scipy import signal


def create_mask(session):
    s2_sceneclassification = session.imagecollection("TERRASCOPE_S2_TOC_V2", bands=["SCENECLASSIFICATION_20M"])

    classification = s2_sceneclassification.band('SCENECLASSIFICATION_20M')

    def makekernel(iwindowsize):
        kernel_vect = scipy.signal.windows.gaussian(iwindowsize, std=iwindowsize / 3.0, sym=True)
        kernel = np.outer(kernel_vect, kernel_vect)
        kernel = kernel / kernel.sum()
        return kernel

    # in openEO, 1 means mask (remove pixel) 0 means keep pixel

    # keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17)) # make small kernel for buffering around pixels whihc belongs not to the suitable classes
    # remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    first_mask = first_mask > 0.057

    # remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10)
    second_mask = second_mask.apply_kernel(makekernel(161)) # bigger kernel for cloud pixels to remove from a larger area pixels
    second_mask = second_mask > 0.1

    return first_mask | second_mask
