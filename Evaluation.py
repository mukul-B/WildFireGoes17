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

import numpy as np
from PIL import Image

from GlobalValues import viirs_dir, goes_dir, compare_dir


# visualising GOES and VIIRS
def viewtiff(v_file, g_file, date, save=False):
    import xarray as xr
    import matplotlib.pyplot as plt
    print(v_file)
    da = xr.open_rasterio(v_file)
    ds = xr.open_rasterio(g_file)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].imshow(da.variable.data[0])
    ax[1].imshow(ds.variable.data[0])
    if (save):
        fig.savefig(f'{compare_dir}{date}.png')
        plt.close()
    plt.show()


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
    gf = np.array(gf)[:, :, 0]
    # (343,)(27, 47)
    print(vf.shape, gf.shape)


def evaluate():
    viirs_list = os.listdir(viirs_dir)
    goes_list = os.listdir(goes_dir)

    for v_file in viirs_list[:4]:
        # g_file = [i for i in goes_list if v_file[5:] in i][0]
        g_file = "GOES" + v_file[5:]
        viewtiff(viirs_dir + v_file, goes_dir + g_file, v_file[6:-4])
        shape_check(viirs_dir + v_file, goes_dir + g_file)
