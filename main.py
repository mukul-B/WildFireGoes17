"""
This script will run through the directory of training images, load
the image pairs

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import pandas as pd

from CommonFunctions import prepareDir
from CreateDataset import createDataset
from Evaluation import evaluate
from GlobalValues import RAD, toExecuteSiteList

if __name__ == '__main__':

    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"]
    for location in locations[:1]:
        prepareDir(location, RAD)
        createDataset(location, product_name=RAD)
        evaluate()

    # viirs_dir = 'data/dixie/VIIRS/'
    # goes_dir = 'data/dixie/GOES/ABI-L1b-RadC/tif/'
    # v_file = 'FIRMS-2021-08-05_854.tif'
    # g_file = 'GOES-2021-08-05_854.tif'
    # viewtiff(viirs_dir + v_file, goes_dir + g_file, '2021-08-05_854')
