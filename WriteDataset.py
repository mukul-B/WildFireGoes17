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

from CommonFunctions import plot_sample
from GlobalValues import viirs_dir, goes_dir


# create training dataset
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0], :]


# def mean_square_iou(targets, inputs):
#     SMOOTH = 1e-6
#     total = targets + inputs
#     intersection = targets * inputs
#     mnse = np.square(targets - inputs)
#     inter = np.square(targets - inputs)
#     tot = np.square(targets - inputs)
#     inter = np.copy(inputs)
#     tot = np.copy(inputs)
#     inter[intersection == 0] = 0
#     tot[total == 0] = 0
#     # TP
#     TP = np.sum(inter)
#     # FP
#
#     # FN
#
#     precision = TP / np.sum(inputs)
#     recall = TP / np.sum(targets)
#
#     # inter[intersection == 0] = 0
#     # tot[total == 0] = 0
#     # union = np.sum(tot) / np.count_nonzero(total)
#     # intersection = np.sum(inter) / np.count_nonzero(intersection)
#     # # IOU = (intersection + SMOOTH) / (union + SMOOTH)
#     # IOU = np.sum(inter)/np.sum(tot)
#     iou = precision*recall / (precision + recall + precision*recall)
#     return precision,recall,iou


#  creating dataset in npy format containing both input and reference files ,
# whole image is croped in window of size 128
def create_training_dataset(v_file, g_file, date, out_dir, location):
    vf = Image.open(v_file)
    gf = Image.open(g_file)
    vf = np.array(vf)[:, :]
    gf = np.array(gf)[:, :]
    # gf = np.array(gf)[:, :, 0]

    if vf.shape != gf.shape:
        print("Write Dataset Failure {}".format(v_file))
        return
    stack = np.stack((vf, gf), axis=2)
    for x, y, window in sliding_window(stack, 128, (128, 128)):
        if window.shape != (128, 128, 2):
            continue
        g_win = window[:, :, 1]
        v_win = window[:, :, 0]
        if np.min(g_win) < 0:
            print("what")
        #  only those windows are considered where it is not mostly empty
        if np.count_nonzero(v_win) == 0 or np.count_nonzero(g_win)==0:

            continue
        else:
            # print()
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
