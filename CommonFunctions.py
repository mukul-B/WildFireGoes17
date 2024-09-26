import math
import os

import numpy as np
from matplotlib import pyplot as plt

from GlobalValues import compare_dir,goes_dir,viirs_dir,logs, testing_dir, training_dir, GOES_ndf,seperate_th


def prepareDir(location, product):
    # site_dir = location
    # product_dir = product
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product['product_name']).replace('$BAND', format(product['band'],'02d'))
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    comp_dir = compare_dir.replace('$LOC', location)

    os.makedirs(goes_tif_dir, exist_ok=True)

    os.makedirs(viirs_tif_dir, exist_ok=True)

    # os.makedirs(comp_dir, exist_ok=True)

    os.makedirs(logs, exist_ok=True)

    os.makedirs(training_dir, exist_ok=True)

    os.makedirs(testing_dir, exist_ok=True)

    os.makedirs(GOES_ndf, exist_ok=True)

    if(seperate_th):
        out_dir_neg = training_dir.replace('training_data','training_data_neg')
        out_dir_TH = training_dir.replace('training_data','training_data_TH')
        out_dir_pos = training_dir.replace('training_data','training_data_pos')

        os.makedirs(out_dir_neg, exist_ok=True)
        os.makedirs(out_dir_TH, exist_ok=True)
        os.makedirs(out_dir_pos, exist_ok=True)

def prepareDirectory(path):
    os.makedirs(path, exist_ok=True)

def plot_sample(col,titles,path=None):
    num = len(col)
    fig, axs = plt.subplots(1, num, constrained_layout=True)
    for i, j in enumerate(col):
        rmse = math.sqrt(np.mean(np.array(((j - col[0]).flatten()) ** 2)))
        p = axs[i].imshow(j)
        # plt.colorbar(p)
        axs[i].set_title(titles[i]+" "+str(rmse))
    if(path):
        plt.savefig(path)
    else:
        plt.show()
    plt.close()