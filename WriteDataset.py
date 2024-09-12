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

from GlobalValues import GOES_Bands, GOES_product_size, viirs_dir, goes_dir, GOES_MIN_VAL, GOES_MAX_VAL, VIIRS_MAX_VAL, training_data_field_names, COLOR_NORMAL_VALUE, seperate_th,th_neg 
import xarray as xr


# create training dataset
# def sliding_window(image, stepSize, windowSize):
#     # slide a window across the image
#     for y in range(0, image.shape[0], stepSize):
#         for x in range(0, image.shape[1], stepSize):
#             # yield the current window
#             yield x, y, image[y:y + windowSize[1], x:x + windowSize[0], :]

def sliding_window(image, stepSize, windowSize):
    """
    Slide a window across the image with full coverage, adjusting the window size at the borders
    and ensuring no part of the image is left uncovered.
    
    Parameters:
    image (ndarray): Input image array.
    stepSize (int): Step size to move the window.
    windowSize (tuple): The (width, height) of the window.
    
    Yields:
    tuple: The top-left corner (x, y) of the window and the window itself.
    """
    img_height, img_width = image.shape[:2]
    
    # Calculate number of steps needed to fully cover the image
    for y in range(0, img_height, stepSize):
        for x in range(0, img_width, stepSize):
            # Adjust the window size at the borders
            x_end = min(x + windowSize[0], img_width)
            y_end = min(y + windowSize[1], img_height)
            
            # Ensure the window covers the border
            x_start = max(0, x_end - windowSize[0])
            y_start = max(0, y_end - windowSize[1])
            
            adjusted_window = image[y_start:y_end, x_start:x_end, :]
            
            yield x_start, y_start, adjusted_window


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
def update_global_max_min(new_value, filename='global_max_min.txt'):
    try:
        # Read existing max and min from the file
        with open(filename, 'r') as file:
            lines = file.readlines()
            global_max = float(lines[0].strip())
            global_min = float(lines[1].strip())
    except FileNotFoundError:
        # If the file does not exist, initialize max and min
        global_max = float('-inf')
        global_min = float('inf')
    except (IndexError, ValueError):
        # If file is empty or contains invalid values, initialize max and min
        global_max = float('-inf')
        global_min = float('inf')

    non_zero = new_value[new_value > 0]
    if non_zero.size > 0:
        min_value = np.min(new_value[new_value > 0])
        # Update global max and min with the new value
        if new_value.max() > global_max:
            global_max = new_value.max()
        if min_value < global_min:
            global_min = min_value

    # Write the updated max and min back to the file
    with open(filename, 'w') as file:
        file.write(f"{global_max}\n")
        file.write(f"{global_min}\n")

    return global_max, global_min


def goes_img_to_channels(gf):
    gf_channels = [None] * GOES_Bands
    # gf_min, gf_max = [210, 207, 205],[413,342, 342]
    gf_min, gf_max = GOES_MIN_VAL, GOES_MAX_VAL
    # gf_min, gf_max = [0,0,0,210, 207],[120,126,126,413,342]
    for i in range(GOES_Bands):
        # if(gf[i].max() == 0):
        if(np.count_nonzero(gf[i]==0) > 0):
            # print(np.count_nonzero(gf[i]==0))
            # print(np.count_nonzero(gf[i]!=0))
            return -1
        if(gf[i].max() > gf_max[i] or gf[i].min() < gf_min[i]):
            print("Error: Max and man values of GOES is incorrect")
            print(i, gf[i].max() , gf_max[i] , gf[i].min() , gf_min[i])
        # update_global_max_min(gf[i],'GOESBAND'+str(i)+'.txt')
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
    if(seperate_th):
        out_dir_neg = out_dir.replace('training_data','training_data_neg')
        out_dir_TH = out_dir.replace('training_data','training_data_TH')
        out_dir_pos = out_dir.replace('training_data','training_data_pos')
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
    gf_min, gf_max = GOES_MIN_VAL[0], GOES_MAX_VAL[0]
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

    if(gf_channels == -1):
        return

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
        if np.count_nonzero(g_win==0) > 0:
            continue

        # based on flag , decide if want empty VIIRS or not
        if(seperate_th):
            frp_win = window[:, :, 4]
            if np.count_nonzero(v_win) == 0:
                np.save(os.path.join(out_dir_neg, 'comb.' + location + '_' + date + '.' + str(x) + '.' + str(y) + '.npy'), window)
            elif np.count_nonzero(v_win) < 60 and np.sum(frp_win) < 600:
                if(th_neg):
                    window[:, :, 0] =0
                np.save(os.path.join(out_dir_TH, 'comb.' + location + '_' + date
                            + '.' + str(x) + '.' + str(y) + '.npy'), window)
            else:
                # with open('fire_frp.txt', 'a') as file:  
                #     file.write(f"{np.count_nonzero(v_win)},{np.sum(frp_win)}\n")  
                np.save(os.path.join(out_dir_pos, 'comb.' + location + '_' + date  + '.' + str(x) + '.' + str(y) + '.npy'), window)

        elif np.count_nonzero(v_win) == 0:
            continue
   
        np.save(os.path.join(out_dir, 'comb.' + location + '_' + date
                        + '.' + str(x) + '.' + str(y) + '.npy'), window)


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
