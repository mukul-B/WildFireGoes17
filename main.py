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
from WriteDataset import writeDataset

if __name__ == '__main__':

    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"]
    product = RAD
    # pipeline run for sites mentioned in toExecuteSiteList
    for location in locations:
        print(location)
        prepareDir(location, product)
        createDataset(location, product_name=product)
        evaluate(location, product)
        writeDataset(location, product)
