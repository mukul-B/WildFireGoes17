"""
This script will have matrix to evalues GOES and VIIRS final dataset
checking dimention of image
checking signal to noise ratio
visualising them

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from GlobalValues import viirs_dir, goes_dir, compare_dir


# visualising GOES and VIIRS
def viewtiff(v_file, g_file, date, save=False):
    # import rasterio
    #
    # raster = rasterio.open(v_file)
    #
    # print(raster.crs.linear_units)
    # exit(0)

    VIIRS_data = xr.open_rasterio(v_file)
    GOES_data = xr.open_rasterio(g_file)
    fig, ax = plt.subplots(1, 3, constrained_layout=True)

    p = ax[0].imshow(VIIRS_data.variable.data[0])
    q = ax[1].imshow(GOES_data.variable.data[0])
    # ploting Viirs on top of GOES
    ax[2].imshow((GOES_data.variable.data[0]) - (VIIRS_data.variable.data[0]))
    plt.colorbar(p, shrink=0.5)
    plt.colorbar(q, shrink=0.5)

    if (save):
        fig.savefig(f'{compare_dir}{date}.png')
        plt.close()
    plt.show()


# create training dataset
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0], :]

#  creating dataset in npy format containing both input and reference files ,
# whole image is croped in window of size 128
def create_training_dataset(v_file, g_file, date, out_dir='data/dixie/training'):
    vf = Image.open(v_file)
    gf = Image.open(g_file)
    vf = np.array(vf)[:, :]
    gf = np.array(gf)[:, :]
    print(vf.dtype)
    if vf.shape != gf.shape:
        print("Failure {}".format(v_file))
        return
    stack = np.stack((vf, gf), axis=2)
    for x, y, window in sliding_window(stack, 128, (128, 128)):
        if window.shape != (128, 128, 2):
            continue
        g_win = window[:, :, 1]
        #  only those windows are considered where it is not mostly empty
        if np.count_nonzero(g_win) / g_win.size < 0.985:
            continue
        else:
            np.save(os.path.join(out_dir, 'comb.' + date
                                 + '.' + str(x) + '.' + str(y) + '.npy'), window)


def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt

    imdff = imdff.flatten()
    rmse = math.sqrt(np.mean(np.array(imdff ** 2)))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def shape_check(v_file, g_file):
    vf = Image.open(v_file)
    gf = Image.open(g_file)
    vf = np.array(vf)[:, :]
    gf = np.array(gf)[:, :]
    # (343,)(27, 47)
    print(vf.shape, gf.shape)
    print(PSNR(gf, vf))

# the dataset created is evaluated visually and statistically
def evaluate(product):
    viirs_list = os.listdir(viirs_dir)
    goes_tif_dir = goes_dir +'/' + product + '/' + 'tif/'
    for v_file in viirs_list:
        g_file = "GOES" + v_file[5:]
        create_training_dataset(viirs_dir + v_file, goes_tif_dir + g_file, v_file[6:-4])
        shape_check(viirs_dir + v_file, goes_tif_dir + g_file)
        viewtiff(viirs_dir + v_file, goes_tif_dir + g_file, v_file[6:-4])
