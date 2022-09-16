"""
This script will run transform the whole image into predicted image by
first converting image in to whole image into , getting prediction and then stiching it back

Created on  sep 15 11:17:09 2022

@author: mukul
"""

import os
from math import ceil

import cartopy.crs as ccrs
import numpy as np
import rasterio
from PIL import Image
from cartopy.io.img_tiles import GoogleTiles
from matplotlib import pyplot as plt
from pyproj import Transformer

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


def plot_improvement2(path):
    print(path)
    transformer = Transformer.from_crs(32611, 4326)
    gfI = Image.open(path)
    gf = np.array(gfI)[:, :, 0]
    with rasterio.open(path, "r") as ds:
        cfl = ds.read(1)
        bl = transformer.transform(ds.bounds.left, ds.bounds.bottom)
        tr = transformer.transform(ds.bounds.right, ds.bounds.top)

    bbox = [bl[1], tr[1], bl[0], tr[0]]
    print(bbox)
    # [-121.10000645737921, -119.90000455967369, 38.39999129409932, 39.59999702266217]
    # [-121,-120,38.5,39.5]

    # data = [(ds.xy(x, y)[0], ds.xy(x, y)[1]) for x, y in np.ndindex(cfl.shape)]
    data = [transformer.transform(ds.xy(x, y)[0], ds.xy(x, y)[1]) for x, y in np.ndindex(cfl.shape)]
    lon = np.array([i[0] for i in data]).reshape(gf.shape)
    lat = np.array([i[1] for i in data]).reshape(gf.shape)
    with open('lat.npy', 'wb') as f:
        np.save(f, lat)
    with open('lon.npy', 'wb') as f:
        np.save(f, lon)
    with open('cfl.npy', 'wb') as f:
        np.save(f, cfl)
    exit(0)
    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
    # axs[0].imshow(gf)
    # axs[1].imshow(cfl)
    # plt.show()
    # 39.59825220841044 38.40175487695621
    # -119.85506345950603 -121.1653760701311
    proj = ccrs.PlateCarree()
    # plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=proj)
    # ax.add_image(StreetmapESRI(), 10)
    # ax.set_extent([-121,-120,38.5,39.5])
    # ax.set_extent(bbox)

    p = ax.pcolormesh(lon, lat, cfl,
                      transform=ccrs.PlateCarree())
    plt.show()

if __name__ == '__main__':
    dir = runtime_dir
    GOES_list = os.listdir(dir)
    for i, gfile in enumerate(GOES_list[:1]):
        plot_improvement2(dir + gfile)
