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
import multiprocessing as mp
from tqdm import tqdm

def on_success(location):
    pbar.update(1)
    print(f"{location} processed successfully at {time.time() - start_time:.2f} seconds")

def on_error(e):
    print(f"Error: {e}")

def CreateDatasetPipeline(location, product, train_test):
    prepareDir(location, product)
    createDataset(location, product)
    evaluate(location, product)
    writeDataset(location, product, train_test)
    return location

if __name__ == '__main__':

    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"]
    # product = RAD
    product = {'product_name': RAD, 'band': 7}
    train_test = training_dir
    start_time = time.time()
    # train_test = testing_dir
    # pipeline run for sites mentioned in toExecuteSiteList
    pool = mp.Pool(8)
    # Initialize tqdm progress bar
    with tqdm(total=len(locations)) as pbar:
        results = []
        for location in locations:
            result = pool.apply_async(CreateDatasetPipeline, args=(location, product, train_test), 
                                      callback=on_success, error_callback=on_error)
            # print(results.get())
            # CreateDatasetPipeline(location, product, train_test)
            results.append(result)
        pool.close()
        pool.join()
        # Print the last location processed
        if results:
            last_processed = results[-1].get()  # Get the result of the last task
            print(f"Last location processed: {last_processed} at {time.time() - start_time:.2f} seconds")
