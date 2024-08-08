"""
This script will run transform the whole image into predicted image by
first converting image in to whole image into , getting prediction and then stiching it back

Created on  sep 15 11:17:09 2022

@author: mukul
"""

import json
import math
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

from AutoEncoderEvaluation import RuntimeDLTransformation
from EvaluationOperation import getth
from GlobalValues import GOES_product_size, RealTimeIncoming_files, RealTimeIncoming_results, GOES_MIN_VAL, GOES_MAX_VAL, VIIRS_MAX_VAL, \
    PREDICTION_UNITS, GOES_UNITS,Results,goes_folder
from LossFunctionConfig import use_config
from RadarProcessing import RadarProcessing
from SiteInfo import SiteInfo
from WriteDataset import goes_img_pkg, goes_img_to_channels, goes_radiance_normaization
import pandas as pd
from datetime import datetime
import xarray as xr



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
        # ro = ro /255
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


def plot_prediction(gpath,output_path,epsg, prediction,supr_resolution):
    
    d = gpath.split('/')[-1].split('.')[0][5:].split('_')
    date_radar = ''.join(d).replace('-', '')
    GOES_data = xr.open_rasterio(gpath)
    gfin = goes_img_pkg(GOES_data)

    if prediction:
        gf_channels = goes_img_to_channels(gfin)
        # Dimensions of the original arrays
        height, width = gfin[0].shape
        # image to model specific window
        window_size = (128,128)
        step_size = 128
        out_channel = 1

        #partion images to  DL model specific window
        partioned_image = partion_image_to_windows(gf_channels,window_size,step_size)
        #applu DL model to windows
        res_image = apply_DLmodel(supr_resolution,partioned_image, window_size)
        #create back image from window output
        reconstructed_gf = reconstruct_from_windows(height, width, res_image, window_size, out_channel)

        pred = reconstructed_gf[0]
        ret1, th1, hist1, bins1, index_of_max_val1 = getth(pred, on=50)
        pred = th1 * pred
        outmap_min = pred.min()
        outmap_max = pred.max()
        pred = (pred - outmap_min) / (outmap_max - outmap_min)
        # pred = retain_adjacent_nonzero(pred)
        pred[pred == 0] = None
        pred = VIIRS_MAX_VAL * pred

    else:
        # gf_min, gf_max = GOES_MIN_VAL, GOES_MAX_VAL
        # gfin = goes_radiance_normaization(gfin, gf_max, gf_min)
        pred = gfin[0]
        # ret1, th1, hist1, bins1, index_of_max_val1 = getth(pred, on=0)
        # pred = th1 * pred
        # pred[pred == 0] = None
        # pred = VIIRS_MAX_VAL * pred

    bbox, lat, lon = get_lon_lat(gpath,epsg)
    proj = ccrs.PlateCarree()
    # plt.style.use('dark_background')
    # plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=proj)
    ax.add_image(StreetmapESRI(), 10)
    # ax.set_extent(bbox)
    cmap = 'YlOrRd'
    # plt.suptitle('Mosquito Fire on {0} at {1} UTC'.format(d[0], d[1]))
    plt.suptitle('{0} at {1}:{2} UTC'.format((datetime.strptime(d[0], '%Y-%m-%d')).strftime('%d %B %Y'), d[1][:2], d[1][2:]))
    p = ax.pcolormesh(lat, lon, pred,
                      transform=ccrs.PlateCarree(),
                    #   vmin=0 if prediction else GOES_MIN_VAL,
                    #   # vmax=34,
                    #   vmax=VIIRS_MAX_VAL if prediction else GOES_MAX_VAL,
                      cmap=cmap)
    
    # geojson_str = latlongTojson(lat,lon,pred)

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
    validate_with_radar = False
    if(validate_with_radar):
        returnval = radarprocessing.plot_radar_json(date_radar, ax)
    # plt.show()
    if (not validate_with_radar) or (returnval):
        # print('/FRP_' + str(d[0] + '_' + d[1]) + '.png')
        result_file =output_path  + str(d[0] + '_' + d[1]) + '.png'
        print(result_file)
        plt.savefig(result_file, bbox_inches='tight', dpi=360)
        # with open(result_file.replace('png','geojson'), "w") as f:
        #     f.write(geojson_str)
    # plt.show()
    plt.close()

def partion_image_to_windows(gf_channels,window_size,step_size):
    partioned_image = {}
    stack = np.stack(gf_channels, axis=2)
    for x, y, window in sliding_window(stack, step_size, window_size):
        partioned_image[(y, x)] = [window[:, :, i] for i in range(stack.shape[2])]
    return partioned_image

def apply_DLmodel(supr_resolution,partioned_image, window_size):
    res_image = {}
    for (x, y), windows in partioned_image.items():
        df = []
        df.append(np.stack(windows, axis=0))
        if(windows[0].shape == window_size):
            
            res_image[(x, y)]  = supr_resolution.Transform(df).numpy()
        else:
            res_image[(x, y)] = windows[0] * 0
    return res_image

def reconstruct_from_windows(height, width, res_image, window_size, out_channel):

    # Initialize empty arrays
    reconstructed_gf = [np.zeros((height, width), dtype=np.float32) for _ in range(out_channel)]
    for (x, y), windows in res_image.items():
            x_end = min(x + window_size[0], reconstructed_gf[0].shape[0])
            y_end = min(y + window_size[1], reconstructed_gf[0].shape[1])
            window_height = x_end - x
            window_width = y_end - y
            reconstructed_gf[0][x:x_end, y:y_end] = windows[:window_height, :window_width]
    return reconstructed_gf

def prePros_ROI(gfin):
    gf_min, gf_max = GOES_MIN_VAL[0], GOES_MAX_VAL[0]
    gfin = goes_radiance_normaization(gfin, gf_max, gf_min)
    gfin = np.nan_to_num(gfin)
    # gfin = gfin.astype(int)
    img = np.round(gfin,5)
    res = image2windows(gfin)
    return res

def retain_adjacent_nonzero(array):
    rows, cols = array.shape
    new_array = np.zeros((rows, cols), dtype=array.dtype)

    for i in range(rows):
        for j in range(cols):
            if array[i, j] != 0:
                # Check if it is an edge
                if i == 0 or i == rows-1 or j == 0 or j == cols-1:
                    new_array[i, j] = array[i, j]
                # Check for adjacent non-zero values
                elif array[i-1, j] != 0 and array[i+1, j] != 0 and array[i, j-1] != 0 and array[i, j+1] != 0 and array[i-1, j-1] != 0 and array[i+1, j+1] != 0 and array[i+1, j-1] != 0 and array[i-1, j+1] != 0:
                    new_array[i, j] = 0
                else:
                    new_array[i, j] = array[i, j]

    return new_array

def latlongTojson(lat,lon,pred):
    features = []
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            if not math.isnan(pred[i, j]):  # Only include points where pred is not NaN
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lat[i, j], lon[i, j]]
                    },
                    "properties": {
                        "pred": float(pred[i, j])  # Ensure pred is a float
                    }
                }
                features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

        # Serialize to JSON
    geojson_str = json.dumps(geojson, indent=2)
    return geojson_str

def get_lon_lat(path,epsg):
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
        os.mkdir(RealTimeIncoming_results+"/"+location)
    if not os.path.exists(RealTimeIncoming_results+"/"+location+"/"+ goes_folder):
        os.mkdir(RealTimeIncoming_results+"/"+location+"/"+goes_folder)
    if not os.path.exists(RealTimeIncoming_results+"/"+location+"/"+"results"):
        os.mkdir(RealTimeIncoming_results+"/"+location+"/"+"results")

def on_success(output_path):
    print(f" processed successfully {output_path}")

def on_error(e):
    print(f"Error: {e}")

if __name__ == '__main__':
    
    data = pd.read_csv(realtimeSiteList)
    locations = data["Sites"]
    plotPredition = True

    supr_resolution = RuntimeDLTransformation(use_config) if plotPredition == True else None
    # mp.set_start_method('spawn', force=True)
    pool = mp.Pool(1)
    
    # pipeline run for sites mentioned in toExecuteSiteList
    prepareDir()
    # implemented only to handle one wildfire event
    # change 1st wildfire location to run for that location
    for location in locations[:1]:
        print(location)
        prepareSiteDir(location)
        site = SiteInfo(location)
        radarprocessing = RadarProcessing(location)
        epsg = site.EPSG
        dir = RealTimeIncoming_files+"/"+location+"/"
        GOES_list = os.listdir(dir)
        # pool = mp.Pool(3)
        pathC = RealTimeIncoming_results +"/"+location + "/"+( "results" if plotPredition else goes_folder) + '/FRP_'
        for gfile in GOES_list:
            
            if not file_exists(pathC + gfile[5:-3] + "png"):
                print(dir + gfile)
                # pool.apply_async(plot_prediction, args=(dir + gfile,pathC,epsg,plotPredition,supr_resolution,), 
                #                       callback=on_success, error_callback=on_error)
                # print(res.get())
                plot_prediction(dir + gfile,pathC,epsg,plotPredition,supr_resolution)
        # pool.close()
        # pool.join()
