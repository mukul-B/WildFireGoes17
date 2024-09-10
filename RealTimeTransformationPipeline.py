"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import os

import pandas as pd
import multiprocessing as mp
from CreateRealtimeDataset import create_realtime_dataset
from GlobalValues import RAD, GOES_product, realtimeSiteList, RealTimeIncoming_files, RealTimeIncoming_results, videos, validate_with_radar




def prepareSiteDir(location):
    goes_tif_dir = RealTimeIncoming_files.replace('$LOC', location)
    os.makedirs(goes_tif_dir, exist_ok=True)


def on_success(output_path):
    print(f" processed successfully {output_path}")

def on_error(e):
    print(f"Error: {e}")

if __name__ == '__main__':

    data = pd.read_csv(realtimeSiteList)
    locations = data["Sites"]
    parallel = 0
    if(parallel):
        pool = mp.Pool(4)
    # pipeline run for sites mentioned in toExecuteSiteList
    # prepareDir()
    # implemented only to handle one wildfire event
    # change 1st wildfire location to run for that location
    for location in locations[:1]:
        print(location)
        prepareSiteDir(location)
        if(parallel):
            ret = pool.apply_async(create_realtime_dataset, args=(location, GOES_product, False, ))
            # print(ret.get())
        else:
            create_realtime_dataset(location, product=GOES_product, verify=False)
    
    if(parallel):
        pool.close()
        pool.join()