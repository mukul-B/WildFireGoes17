"""
This script will run transform the whole image into predicted image by
first converting image in to whole image into , getting prediction and then stiching it back

Created on  sep 15 11:17:09 2022

@author: mukul
"""

import os
from math import ceil

import numpy as np
from PIL import Image
from cartopy.io.img_tiles import GoogleTiles
from matplotlib import pyplot as plt

from AutoEncoder.eval import supr_resolution
from GlobalValues import runtime_dir


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:  # <-- the only change (0 indicates no padding)
        vector[-pad_width[1]:] = pad_value


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def image2windows(gf):
    h = ceil(len(gf) / 128)
    w = ceil(len(gf[0]) / 128)
    # result = [[0] * h] * w
    result = [[np.array((0, 0)) for x in range(h)] for y in range(w)]
    for x, y, window in sliding_window(gf, 128, (128, 128)):
        result[int(x / 128)][int(y / 128)] = window
    return np.array(result)


def padWindow(window):
    sx, sy = window.shape
    padx, pady = 128 - sx, 128 - sy
    ro = np.pad(window, ((0, padx), (0, pady)), pad_with, padder=0)
    ro = supr_resolution([ro])
    ro = ro[0:sx, 0:sy]
    return ro


def windows2image(windows):
    full = np.empty((0, 0), int)
    for i in range(len(windows)):
        row = padWindow(windows[0][i])
        for j in range(1, len(windows[0])):
            ro = windows[j][i]
            ro = padWindow(ro)
            row = np.hstack((row, ro))

        if full.shape == (0, 0):
            full = row
        else:
            full = np.vstack((full, row))

    return full


class StreetmapESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Street_Map/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url


def plot_improvement(path='reference_data/Dixie/GOES/ABI-L1b-RadC/tif/GOES-2021-07-16_1012.tif'):
    d = path.split('/')[-1].split('.')[0][5:].split('_')
    gfI = Image.open(path)
    gf = np.array(gfI)[:, :, 0]
    res = image2windows(gf)
    gf2 = windows2image(res)
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle('Mosquito Fire on ' + d[0] + ' at ' + d[1])
    axs[0].imshow(gf)
    axs[0].set_title('Original GOES')
    axs[1].imshow(gf2)
    axs[1].set_title('Super_Resolution GOES')
    plt.savefig('result_for_video' + '/FRP_' + str(d[0] + '_' + d[1]) + '.png', bbox_inches='tight', dpi=240)


if __name__ == '__main__':
    dir = runtime_dir
    GOES_list = os.listdir(dir)
    for i, gfile in enumerate(GOES_list):
        plot_improvement(dir + gfile)