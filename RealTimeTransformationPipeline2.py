"""
This script will run transform the whole image into predicted image by
first converting image in to whole image into , getting prediction and then stiching it back

Created on  sep 15 11:17:09 2022

@author: mukul
"""


import multiprocessing as mp
import os

from pandas.io.common import file_exists
from AutoEncoderEvaluation import RuntimeDLTransformation
from CreateRealtimeDataset import create_realtime_dataset
from ModelRunConfiguration import use_config
from RadarProcessing import RadarProcessing
from RealTimeTransformation import plot_prediction
from SiteInfo import SiteInfo
import pandas as pd
from GlobalValues import GOES_product, realtimeSiteList
from GlobalValues import  RealTimeIncoming_files, RealTimeIncoming_results, goes_folder



def prepareSiteDir(location):
    results = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE', "results" )
    input_plot = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE', goes_folder)
    goes_tif_dir = RealTimeIncoming_files.replace('$LOC', location)
    os.makedirs(goes_tif_dir, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    os.makedirs(input_plot, exist_ok=True)


def on_success(output_path):
    if(output_path != None):
        print(f" processed successfully {output_path}")

def on_error(e):
    print(f"Error: {e}")

if __name__ == '__main__':

    data = pd.read_csv(realtimeSiteList)
    locations = data["Sites"]
    plotPredition = True
    supr_resolution = RuntimeDLTransformation(use_config) if plotPredition == True else None

    parallel = 0
    if(parallel):
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(4)
    # pipeline run for sites mentioned in toExecuteSiteList

    # implemented only to handle one wildfire event
    # change 1st wildfire location to run for that location
    for location in locations[:1]:
        print(location)
        dir =RealTimeIncoming_files.replace('$LOC', location)
        pathC = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE', "results" if plotPredition else goes_folder)
        prepareSiteDir(location)

        site = SiteInfo(location)
        radarprocessing = RadarProcessing(location)
        epsg = site.EPSG
        create_realtime_dataset(location, product=GOES_product, verify=False)

        GOES_list = os.listdir(dir)
        for gfile in GOES_list:
            
            # if not file_exists(pathC + gfile[5:-3] + "png"):
                if(parallel):
                    pool.apply_async(plot_prediction, args=(dir + gfile,pathC,epsg,plotPredition,supr_resolution,), 
                                        callback=on_success, error_callback=on_error)
                    # print(res.get())
                else:
                    plot_prediction(dir + gfile,pathC,epsg,plotPredition,supr_resolution)
    if(parallel):
        pool.close()
        pool.join()