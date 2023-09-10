"""
This script will run transform the whole image into predicted image by
first converting image in to whole image into , getting prediction and then stiching it back

Created on  sep 15 11:17:09 2022

@author: mukul
"""

import multiprocessing as mp
import os
from math import ceil

import cartopy.crs as ccrs
import numpy as np
import rasterio
from PIL import Image
from cartopy.io.img_tiles import GoogleTiles
from matplotlib import pyplot as plt
from pandas.io.common import file_exists
from pyproj import Transformer

from AutoEncoderEvaluation import supr_resolution
from EvaluationMetrics import getth
from GlobalValues import RealTimeIncoming_files, RealTimeIncoming_results, GOES_MIN_VAL, GOES_MAX_VAL, VIIRS_MAX_VAL, \
    PREDICTION_UNITS, GOES_UNITS,Results,goes_folder
from LossFunctionConfig import use_config
from RadarProcessing import RadarProcessing
from SiteInfo import SiteInfo
from WriteDataset import goes_radiance_normaization
import pandas as pd


from GlobalValues import realtimeSiteList, RealTimeIncoming_files, RealTimeIncoming_results


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
    return result


def padWindow(window):
    sx, sy = window.shape
    padx, pady = 128 - sx, 128 - sy
    ro = window
    # ro = np.pad(window, ((0, padx), (0, pady)), pad_with, padder=0)
    if ro.shape != (128, 128):
        ro = np.zeros(window.shape)
    else:
        ro = supr_resolution(use_config, [ro])
        # ro = ro
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
    # res = image2windows(gf)
    # gf2 = windows2image(res)
    gf2 = gf
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle('Mosquito Fire on ' + d[0] + ' at ' + d[1])
    axs[0].imshow(gf)
    axs[0].set_title('Original GOES')
    axs[1].imshow(gf2)
    axs[1].set_title('Super_Resolution GOES')

    plt.savefig('result_for_video' + '/FRP_' + str(d[0] + '_' + d[1]) + '.png', bbox_inches='tight', dpi=240)


def plot_prediction(gpath,output_path,epsg, prediction=True):
    print(gpath,output_path)
    # return 0
    d = gpath.split('/')[-1].split('.')[0][5:].split('_')
    print(d)
    date_radar = ''.join(d).replace('-', '')
    gfI = Image.open(gpath)
    gfin = np.array(gfI)[:, :]
    # gfin = np.array(gfI)[:, :, 0]

    if prediction:
        # gfin_min , gfin_max = np.min(gfin) , np.max(gfin)
        gf_min, gf_max = GOES_MIN_VAL, GOES_MAX_VAL
        gfin = goes_radiance_normaization(gfin, gf_max, gf_min)
        gfin = np.nan_to_num(gfin)
        gfin = gfin.astype(int)
        res = image2windows(gfin)
        pred = windows2image(res)
        # modelPrediction = ModelPrediction4singleEvent(use_config)
        # pred = modelPrediction.prediction(gfin)
        ret1, th1, hist1, bins1, index_of_max_val1 = getth(pred, on=0)
        pred = th1 * pred
        pred[pred == 0] = None
        pred = VIIRS_MAX_VAL * pred
        

    else:
        pred = gfin
        

    bbox, lat, lon = get_lon_lat(gpath,epsg)
    proj = ccrs.PlateCarree()
    # plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=proj)
    ax.add_image(StreetmapESRI(), 10)
    # ax.set_extent(bbox)
    cmap = 'YlOrRd'
    # plt.suptitle('Mosquito Fire on {0} at {1} UTC'.format(d[0], d[1]))
    plt.suptitle('{0} at {1}:{2} UTC'.format(d[0], d[1][:2], d[1][2:]))
    p = ax.pcolormesh(lat, lon, pred,
                      transform=ccrs.PlateCarree(),
                      vmin=0,
                      # vmax=34,
                      vmax=GOES_MAX_VAL,
                      cmap=cmap)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, alpha=0.5)
    cb = plt.colorbar(p, pad=0.01)
    cb.ax.tick_params(labelsize=11)
    cb.set_label(PREDICTION_UNITS if prediction else GOES_UNITS, fontsize=12)
    gl.top_labels = False
    gl.right_labels = False
    # gl.xlines = False
    # gl.ylines = False
    gl.xlabel_style = {'size': 9, 'rotation': 30}
    gl.ylabel_style = {'size': 9}
    plt.tight_layout()
    radarprocessing = RadarProcessing()
    # returnval = radarprocessing.plot_radar_json(f'radar_data/Bear/bear_{date_radar}_smooth_perim.geojson', ax)
    returnval = radarprocessing.plot_radar_json(f'radar_data/Caldor/Caldor_{date_radar}_smooth_perim_new.geojson', ax)
    # plt.show()
    if returnval:
        print('/FRP_' + str(d[0] + '_' + d[1]) + '.png')
        plt.savefig(output_path  + str(d[0] + '_' + d[1]) + '.png', bbox_inches='tight', dpi=240)
    # plt.show()
    plt.close()


def get_lon_lat(path,epsg):
    print(epsg)
    # caldor 32611
    # bear 32610
    transformer = Transformer.from_crs(epsg, 4326)
    with rasterio.open(path, "r") as ds:
        cfl = ds.read(1)
        bl = transformer.transform(ds.bounds.left, ds.bounds.bottom)
        tr = transformer.transform(ds.bounds.right, ds.bounds.top)
    bbox = [bl[1], tr[1], bl[0], tr[0]]
    data = [transformer.transform(ds.xy(x, y)[0], ds.xy(x, y)[1]) for x, y in np.ndindex(cfl.shape)]
    lon = np.array([i[0] for i in data]).reshape(cfl.shape)
    lat = np.array([i[1] for i in data]).reshape(cfl.shape)
    return bbox, lat, lon


def prepareDir():
    if not os.path.exists(RealTimeIncoming_files):
        os.mkdir(RealTimeIncoming_files)
    if not os.path.exists(RealTimeIncoming_results):
        os.mkdir(RealTimeIncoming_results)

def prepareSiteDir(location):
    if not os.path.exists(RealTimeIncoming_results+"/"+location):
        print(RealTimeIncoming_results+"/"+location)
        os.mkdir(RealTimeIncoming_results+"/"+location)
    if not os.path.exists(RealTimeIncoming_results+"/"+location+"/"+ goes_folder):
        print(RealTimeIncoming_results+"/"+location+"/"+ goes_folder)
        os.mkdir(RealTimeIncoming_results+"/"+location+"/"+goes_folder)
    if not os.path.exists(RealTimeIncoming_results+"/"+location+"/"+Results):
        print(RealTimeIncoming_results+"/"+location+"/"+ Results)
        os.mkdir(RealTimeIncoming_results+"/"+location+"/"+Results)


if __name__ == '__main__':


    print(realtimeSiteList)
    data = pd.read_csv(realtimeSiteList)
    locations = data["Sites"]
    plotPredition = True
    # pool = mp.Pool(8)
    # pipeline run for sites mentioned in toExecuteSiteList
    prepareDir()
    # implemented only to handle one wildfire event
    # change 1st wildfire location to run for that location
    for location in locations[:1]:
        print(location)
        prepareSiteDir(location)
        site = SiteInfo(location)
        epsg = site.EPSG
        dir = RealTimeIncoming_files+"/"+location+"/"
        GOES_list = os.listdir(dir)
        print(GOES_list)
        pool = mp.Pool(6)
        pathC = RealTimeIncoming_results +"/"+location + "/"+( Results if plotPredition else goes_folder) + '/FRP_'
        for gfile in GOES_list:
            print(dir + gfile)
            if not file_exists(pathC + gfile[5:-3] + "png"):
                # pool.apply_async(plot_prediction, args=(dir + gfile,))
                # print(res.get())
                plot_prediction(dir + gfile,pathC,epsg,plotPredition)
        pool.close()
        pool.join()





    
