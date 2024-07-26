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

from GlobalValues import GOES_Bands, GOES_product_size, viirs_dir, goes_dir, GOES_MIN_VAL, GOES_MAX_VAL, VIIRS_MAX_VAL, training_data_field_names, COLOR_NORMAL_VALUE
import xarray as xr


# create training dataset
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0], :]


def viirs_radiance_normaization(vf, vf_max):
    color_normal_value = COLOR_NORMAL_VALUE
    if vf_max > 1:
        return color_normal_value * (vf / vf_max)
    return vf


def goes_radiance_normaization(gf, gf_max, gf_min):
    color_normal_value = COLOR_NORMAL_VALUE

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

def goes_img_pkg(GOES_data):
    gf = [None] * GOES_product_size
    for i in range(GOES_product_size):
        gf[i] = GOES_data.variable.data[i]
        gf[i] = np.array(gf[i])[:, :]

        # gf[i] = Normalize_img(gf[i])
    return gf

# def goes_img_to_channels(gf):
#     gf_channels = [None] * GOES_Bands
#     gf_channels[0] = gf[0]
#     gf_channels[0] = Normalize_img(gf_channels[0])

#     Active_fire = (gf[0]-gf[1])/(gf[0]+gf[1])
#     cloud_remove_280 = Active_fire * (gf[2]> 280) * 1000
#     cloud_remove = cloud_remove_280 * (cloud_remove_280 > 0) 
#     cloud_remove = (cloud_remove * GOES_MAX_VAL)  / cloud_remove.max()
#     gf_channels[1] = Normalize_img(cloud_remove,gf_min = 0, gf_max = GOES_MAX_VAL)
#     return gf_channels

def goes_img_to_channels(gf):
    gf_channels = [None] * GOES_Bands
    gf_min, gf_max = [210, 207, 205],[413,342, 342]
    for i in range(GOES_Bands):
        # gf_min, gf_max = np.min(gf[i]), np.max(gf[i])
        # gf_min, gf_max = GOES_MIN_VAL, GOES_MAX_VAL
        
        
        # non_zero = gf[i][gf[i] !=0]
        # if non_zero.size > 0:
        #     if np.min(gf[i]) < global_min[i]:
        #         global_min[i] = np.min(gf[i][gf[i] !=0])
             
        # if np.max(gf[i]) > global_max[i]:
        #     global_max[i] = np.max(gf[i])

        # if(non_zero.size == 0 and np.max(gf[i]) > 0 ):
        #     print("Issue-----------------------------------------------------")
        #[210.16818, 207.01837, 205.42421] [412.24646, 341.23163, 341.24542]

        gf_channels[i] = Normalize_img(gf[i],gf_min[i], gf_max[i])
    
    return gf_channels

def Normalize_img(img,gf_min = GOES_MIN_VAL, gf_max = GOES_MAX_VAL):
    img2 = goes_radiance_normaization(img, gf_max, gf_min)
    img = np.nan_to_num(img2)
    # img = img.astype(int)
    img = np.round(img,5)
    # diff = np.max(img2-img)
    # actual = diff * (gf_max - gf_min)
    # if(actual > 0.002):
    #     print(actual)
    return img


#  creating dataset in npy format containing both input and reference files ,
# whole image is croped in window of size 128
def create_training_dataset(v_file, g_file, date, out_dir, location):
    td = {}
    out_dir_neg = out_dir.replace('classifier','classifier_neg')
    out_dir_pos = out_dir.replace('classifier','classifier_pos')
    # vf = Image.open(v_file)
    VIIRS_data = xr.open_rasterio(v_file)
    vf = VIIRS_data.variable.data[0]
    GOES_data = xr.open_rasterio(g_file)
    kf = goes_img_pkg(GOES_data)
    gf = GOES_data.variable.data[0]
    # gf = Image.open(g_file)
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
    # gf = goes_radiance_normaization(gf, gf_max, gf_min)
    # gf = np.nan_to_num(gf)
    # gf = gf.astype(int)

    # vf_max = np.max(vf)
    vf_max = VIIRS_MAX_VAL
    vf = viirs_radiance_normaization(vf, vf_max)
    # vf = vf.astype(int)
    vf = np.round(vf,5)

    gf_min = np.full(gf.shape, gf_min)
    gf_max = np.full(gf.shape, gf_max)
    vf_max = np.full(gf.shape, vf_max)
    training_data_with_field = {
    'vf': vf,
    'vf_FRP': vf_FRP,
    'gf_min': gf_min,
    'gf_max': gf_max,
    'vf_max': vf_max
}
    
    gf_channels = goes_img_to_channels(kf)

    for i in range(GOES_Bands):
        training_data_with_field[f'gf_c{i+1}'] = gf_channels[i]

    ordered_data = [training_data_with_field[key] for key in training_data_field_names]

    if(len(ordered_data) != len(training_data_with_field)):
        print("not matching expecting fields for training data")
    stack = np.stack(ordered_data, axis=2)
    # stack = np.stack((vf, kf[0], kf[1],kf[2], vf_FRP, gf_min, gf_max, vf_max), axis=2)
    for x, y, window in sliding_window(stack, 128, (128, 128)):
        if window.shape != (128, 128, len(training_data_field_names)):
            continue
        g_win = window[:, :, 1]
        v_win = window[:, :, 0]
        vf_win = window[:, :, 2]
        #  only those windows are considered where it is not mostly empty
        if np.count_nonzero(v_win) == 0 or np.count_nonzero(g_win) == 0:
            # if np.count_nonzero(v_win) == 0:
                # pass
                # np.save(os.path.join(out_dir_neg, 'comb.' + location + '_' + date
                #                  + '.' + str(x) + '.' + str(y) + '.npy'), window)

            continue
        else:
            # print(location + '_' + date,g_win.sum()/float(255),v_win.sum()/float(255))
            np.save(os.path.join(out_dir_pos, 'comb.' + location + '_' + date
                                 + '.' + str(x) + '.' + str(y) + '.npy'), window)
        # np.save(os.path.join(out_dir, 'comb.' + location + '_' + date
        #                         + '.' + str(x) + '.' + str(y) + '.npy'), window)      


def writeDataset(location, product, train_test):
    # product_name = product['product_name']
    # band = product['band']
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product['product_name']).replace('$BAND', format(product['band'],'02d'))
    viirs_list = os.listdir(viirs_tif_dir)
    # global global_max, global_min
    # global_max = [0] * GOES_Bands
    # global_min = [500] * GOES_Bands
    
    for v_file in viirs_list:
        g_file = "GOES" + v_file[10:]
        create_training_dataset(viirs_tif_dir + v_file, goes_tif_dir + g_file, v_file[11:-4],
                                out_dir=train_test, location=location)
    
    # return global_min, global_max
