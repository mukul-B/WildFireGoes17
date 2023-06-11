"""
This script will have matrix to evalues GOES and VIIRS final dataset
checking dimention of image
checking signal to noise ratio
visualising them

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import os

import numpy as np
from PIL import Image

from GlobalValues import viirs_dir, goes_dir, GOES_MIN_VAL, GOES_MAX_VAL, VIIRS_MAX_VAL
import xarray as xr


# create training dataset
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0], :]


def viirs_radiance_normaization(vf, vf_max):
    color_normal_value = 255
    if vf_max > 1:
        return color_normal_value * (vf / vf_max)
    return vf


def goes_radiance_normaization(gf, gf_max, gf_min):
    color_normal_value = 255

    # if(goes_scene[layer].values.max()>=413 or  goes_scene[layer].values.min()<210):
    #     print(goes_scene[layer].values.max(), goes_scene[layer].values.min())
    # goes_scene[layer].values = np.nan_to_num(goes_scene[layer].values)
    # if goes_scene[layer].values.max() == 0:
    #     return -1
    # if (goes_scene[layer].values.max() >= 413 or goes_scene[layer].values.min() < 210):
    #     print(goes_scene[layer].values.max(), goes_scene[layer].values.min())
    # goes_scene[layer].values = 255 * (goes_scene[layer].values - goes_scene[layer].values.min()) / (
    #         goes_scene[layer].values.max() - goes_scene[layer].values.min())
    # goes_scene[layer].values = 255 * goes_scene[layer].values / 413
    # goes_scene[layer].values = np.nan_to_num(goes_scene[layer].values)
    #
    return color_normal_value * ((gf - gf_min) / (gf_max - gf_min))


#  creating dataset in npy format containing both input and reference files ,
# whole image is croped in window of size 128
def create_training_dataset(v_file, g_file, date, out_dir, location):
    # vf = Image.open(v_file)
    VIIRS_data = xr.open_rasterio(v_file)
    vf = VIIRS_data.variable.data[0]
    gf = Image.open(g_file)
    vf = np.array(vf)[:, :]
    gf = np.array(gf)[:, :]
    # gf = np.array(gf)[:, :, 0]

    if vf.shape != gf.shape:
        print("Write Dataset Failure {}".format(v_file))
        return

    vf_FRP = VIIRS_data.variable.data[1]
    vf_FRP = np.array(vf_FRP)[:, :]

    # gf_min, gf_max = np.min(gf), np.max(gf)
    gf_min, gf_max = GOES_MIN_VAL, GOES_MAX_VAL
    gf = goes_radiance_normaization(gf, gf_max, gf_min)
    gf = np.nan_to_num(gf)
    gf = gf.astype(int)

    # vf_max = np.max(vf)
    vf_max = VIIRS_MAX_VAL
    vf = viirs_radiance_normaization(vf, vf_max)
    vf = vf.astype(int)

    gf_min = np.full(gf.shape, gf_min)
    gf_max = np.full(gf.shape, gf_max)
    vf_max = np.full(gf.shape, vf_max)

    stack = np.stack((vf, gf, vf_FRP, gf_min, gf_max, vf_max), axis=2)
    for x, y, window in sliding_window(stack, 128, (128, 128)):
        if window.shape != (128, 128, 6):
            continue
        g_win = window[:, :, 1]
        v_win = window[:, :, 0]
        vf_win = window[:, :, 2]
        #  only those windows are considered where it is not mostly empty
        if np.count_nonzero(v_win) == 0 or np.count_nonzero(g_win) == 0:
            continue
        else:
            np.save(os.path.join(out_dir, 'comb.' + location + '_' + date
                                 + '.' + str(x) + '.' + str(y) + '.npy'), window)


def writeDataset(location, product, train_test):
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product)
    viirs_list = os.listdir(viirs_tif_dir)
    for v_file in viirs_list:
        g_file = "GOES" + v_file[5:]
        create_training_dataset(viirs_tif_dir + v_file, goes_tif_dir + g_file, v_file[6:-4],
                                out_dir=train_test, location=location)
