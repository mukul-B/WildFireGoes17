"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import os

import pandas as pd

from CreateRealtimeDataset import create_realtime_dataset
from GlobalValues import RAD, GOES_product, realtimeSiteList, RealTimeIncoming_files, RealTimeIncoming_results, videos, validate_with_radar


def prepareDir():
    if not os.path.exists(RealTimeIncoming_files):
        os.mkdir(RealTimeIncoming_files)
    if not os.path.exists(RealTimeIncoming_results):
        os.mkdir(RealTimeIncoming_results)
    if not os.path.exists(videos):
        os.mkdir(videos)

def prepareSiteDir(location):
    if not os.path.exists(RealTimeIncoming_files+"/"+location):
        os.mkdir(RealTimeIncoming_files+"/"+location)
    if not os.path.exists(RealTimeIncoming_results+"/"+location):
        os.mkdir(RealTimeIncoming_results+"/"+location)

if __name__ == '__main__':

    print(realtimeSiteList)
    data = pd.read_csv(realtimeSiteList)
    locations = data["Sites"]
    product = RAD
    # pool = mp.Pool(8)
    # pipeline run for sites mentioned in toExecuteSiteList
    prepareDir()
    # implemented only to handle one wildfire event
    # change 1st wildfire location to run for that location
    for location in locations[:1]:
        print(location)
        prepareSiteDir(location)
        # ret = pool.apply_async(create_realtime_dataset, args=(product,))
        # print(ret.get())
        create_realtime_dataset(location, product=GOES_product, verify=False)
    # pool.close()
    # pool.join()