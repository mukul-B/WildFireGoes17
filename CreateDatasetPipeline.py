"""
This script will run through the directory of training images, load
the image pairs

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import os
import pandas as pd

from CommonFunctions import prepareDir
from CreateTrainingDataset import createDataset
from ValidateAndVisualizeDataset import validateAndVisualizeDataset
from WriteDataset import writeDataset
from GlobalValues import RAD, GOES_product, toExecuteSiteList, training_dir,testing_dir, realtimeSiteList
import time
import multiprocessing as mp
from tqdm import tqdm

def on_success(location):
    pbar.update(1)
    print(f"{location} processed successfully at {time.time() - start_time:.2f} seconds")

def on_error(e):
    print(f"Error: {e}")

def CreateDatasetPipeline(location, product, train_test):
    print(location)
    prepareDir(location, product)
    createDataset(location, product)
    validateAndVisualizeDataset(location, product)
    writeDataset(location, product, train_test)
    
    return location

def count_training_set_created(dir):
    files_and_dirs = os.listdir(dir)

    try:
        file_list_pos = os.listdir(dir.replace('training_data','training_data_pos'))
        print(f'file_list_pos: {len(file_list_pos)}')
        file_list_neg = os.listdir(dir.replace('training_data','training_data_neg'))
        print(f'file_list_neg: {len(file_list_neg)}')
        file_list_TH = os.listdir(dir.replace('training_data','training_data_TH'))
        print(f'file_list_TH: {len(file_list_TH)}')
    except:
        print("split_incomplete")
    # Count only the files (not directories)
    file_count = sum(os.path.isfile(os.path.join(dir, item)) for item in files_and_dirs)

    print(f'{file_count} records writtenin {dir}')

if __name__ == '__main__':
    print(f'Creating dataset for sites in {toExecuteSiteList}')
    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"][:]
    train_test = training_dir
    start_time = time.time()
    # train_test = testing_dir
    # pipeline run for sites mentioned in toExecuteSiteList
    
    parallel = 1
    if(parallel):
        pool = mp.Pool(8)
    # Initialize tqdm progress bar
    with tqdm(total=len(locations)) as pbar:
        results = []
        for location in locations:
            if(parallel):
                result = pool.apply_async(CreateDatasetPipeline, args=(location, GOES_product, train_test), 
                                      callback=on_success, error_callback=on_error)
                # print(results.get())
            else:
                result = CreateDatasetPipeline(location, GOES_product, train_test)
                on_success(location)
            # results.append(result)
        if(parallel):
            pool.close()
            pool.join()
        # Print the last location processed
        # if results:
        #     last_processed = results[-1].get()  # Get the result of the last task
        #     print(f"Last location processed: {last_processed} at {time.time() - start_time:.2f} seconds")
    count_training_set_created(train_test)
    
