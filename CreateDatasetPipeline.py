"""
This script will run through the directory of training images, load
the image pairs

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import pandas as pd

from CommonFunctions import prepareDir
from CreateDataset import createDataset
from EvaluationDataset import evaluate
from WriteDataset import writeDataset
from GlobalValues import RAD, toExecuteSiteList, training_dir,testing_dir, realtimeSiteList
import time


if __name__ == '__main__':

    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"]
    # product = RAD
    product = {'product_name': RAD, 'band': 7}
    train_test = training_dir
    start_time = time.time()
    # train_test = testing_dir
    # pipeline run for sites mentioned in toExecuteSiteList
    # pool = mp.Pool(8)
    for location in locations:
        print(location)
        # ret = pool.apply_async(create_realtime_dataset, args=(product,))
        # print(ret.get())
        prepareDir(location, product)
        createDataset(location, product=product)
        # print("--- %s seconds ---" % (time.time() - start_time))
        evaluate(location, product)
        writeDataset(location, product, train_test)
    # pool.close()
    # pool.join()
