import math
import os

import numpy as np
from matplotlib import pyplot as plt

from GlobalValues import compare_dir,goes_dir,viirs_dir,data_dir, goes_folder, viirs_folder, compare, logs, testing_dir, training_dir, GOES_ndf


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

    os.makedirs(comp_dir, exist_ok=True)


    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # if not os.path.exists(data_dir + '/' + site_dir):
    #     os.mkdir(data_dir + '/' + site_dir)
    # if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder):
    #     os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder)
    # if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir):
    #     os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir)
    # if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir + '/tif'):
    #     os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir + '/tif')

    # if not os.path.exists(data_dir + '/' + site_dir + '/' + viirs_folder):
    #     os.mkdir(data_dir + '/' + site_dir + '/' + viirs_folder)

    # if not os.path.exists(data_dir + '/' + site_dir + '/' + compare):
    #     os.mkdir(data_dir + '/' + site_dir + '/' + compare)

    if not os.path.exists(logs):
        os.mkdir(logs)
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)

    if not os.path.exists(testing_dir):
        os.mkdir(testing_dir)

    if not os.path.exists(GOES_ndf):
        os.mkdir(GOES_ndf)


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