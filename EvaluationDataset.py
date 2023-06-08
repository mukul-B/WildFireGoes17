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
import datetime

# demonstrate reference_data standardization with sklearn


# visualising GOES and VIIRS
def viewtiff(v_file, g_file, date, save=True, compare_dir=None):
    VIIRS_data = xr.open_rasterio(v_file)
    GOES_data = xr.open_rasterio(g_file)

    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))

    vd = VIIRS_data.variable.data[0]
    gd = GOES_data.variable.data[0]

    X, Y = np.mgrid[0:1:complex(str(vd.shape[0]) + "j"), 0:1:complex(str(vd.shape[1]) + "j")]
    # plot_individual_images(X, Y, compare_dir, g_file, gd, vd)
    # if in GOES and VIIRS , the values are normalized, using this flag to visualize result
    normalized = False
    vmin,vmax = (0, 250) if normalized else (200,420)
    p = ax[1].pcolormesh(Y, -X, vd, cmap="jet", vmin=vmin, vmax=vmax)
    q = ax[0].pcolormesh(Y, -X, gd, cmap="jet", vmin=vmin, vmax=vmax)
    r = ax[2].pcolormesh(Y, -X, (gd - vd), cmap="jet", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(p, pad=0.01)
    cb.ax.tick_params(labelsize=11)
    cb.set_label('Radiance (K)', fontsize=12)
    for k in range(3):
        ax[k].tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
    plt.rcParams['savefig.dpi'] = 600
    if (save):
        fig.savefig(f'{compare_dir}{date}.png')
        plt.close()
    plt.show()


def plot_individual_images(X, Y, compare_dir, g_file, gd, vd):
    ar = [vd, gd, gd - vd]
    if (g_file == 'reference_data/Kincade/GOES/ABI-L1b-RadC/tif/GOES-2019-10-27_949.tif'):
        print(g_file, compare_dir)
        for k in range(3):
            fig2 = plt.figure()
            ax = fig2.add_subplot()
            a = ax.pcolormesh(Y, -X, ar[k], cmap="jet", vmin=200, vmax=420)
            cb = fig2.colorbar(a, pad=0.01)
            cb.ax.tick_params(labelsize=11)
            cb.set_label('Radiance (K)', fontsize=12)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            # plt.show()
            plt.savefig(f'{compare_dir}/data_preprocessing{k}.png', bbox_inches='tight', dpi=600)
            plt.close()



def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    print(imdff)

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
def evaluate(location, product):
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product)
    comp_dir = compare_dir.replace('$LOC', location)
    viirs_list = os.listdir(viirs_tif_dir)
    for v_file in viirs_list:
        g_file = "GOES" + v_file[5:]
        # print(g_file)
        # shape_check(viirs_tif_dir + v_file, goes_tif_dir + g_file)
        viewtiff(viirs_tif_dir + v_file, goes_tif_dir + g_file, v_file[6:-4], compare_dir=comp_dir, save=True)
