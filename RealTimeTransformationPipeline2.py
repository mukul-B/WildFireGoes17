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
from ModelRunConfiguration import real_time_config
from RadarProcessing import RadarProcessing
from RealTimeTransformation import plot_prediction
from SiteInfo import SiteInfo
import pandas as pd
from GlobalValues import GOES_product, realtimeSiteList
from GlobalValues import  RealTimeIncoming_files, RealTimeIncoming_results, goes_folder, viirs_folder



def prepareSiteDir(location):
    results = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE', "results" )
    results_VIIRS = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE', viirs_folder )
    input_plot = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE', goes_folder)
    goes_tif_dir = RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', goes_folder)
    goes_test_tif_dir = RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', goes_folder+'test')
    VIIRS_tif_dir = RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', viirs_folder)
    os.makedirs(VIIRS_tif_dir, exist_ok=True)
    os.makedirs(goes_tif_dir, exist_ok=True)
    os.makedirs(goes_test_tif_dir, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    os.makedirs(results_VIIRS, exist_ok=True)
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
    VIIRS_MATCH = True
    supr_resolution = RuntimeDLTransformation(real_time_config) if plotPredition == True else None

    parallel = 1
    if(parallel):
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(4)
    # pipeline run for sites mentioned in toExecuteSiteList

    # implemented only to handle one wildfire event
    # change 1st wildfire location to run for that location
    for location in locations[:1]:
        print(location)
        dir =RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', goes_folder if VIIRS_MATCH == False else goes_folder+'test')
        result_folder = "results" if plotPredition else goes_folder
        if(VIIRS_MATCH):
            result_folder = viirs_folder
            VIIRS_dir =RealTimeIncoming_files.replace('$LOC', location).replace('$RESULT_TYPE', viirs_folder)
        else:
            VIIRS_dir = None

        pathC = RealTimeIncoming_results.replace('$LOC', location).replace('$RESULT_TYPE',result_folder )
        prepareSiteDir(location)

        site = SiteInfo(location)
        radarprocessing = RadarProcessing(location)
        epsg = site.EPSG
        create_realtime_dataset(location, product=GOES_product, verify=False,validate_with_VIIRS=VIIRS_MATCH)

        GOES_list = os.listdir(dir)
        for gfile in GOES_list:
            
            # if not file_exists(pathC + gfile[5:-3] + "png"):
                if(parallel):
                    pool.apply_async(plot_prediction, args=(dir + gfile,pathC,epsg,plotPredition,supr_resolution,VIIRS_dir,), 
                                        callback=on_success, error_callback=on_error)
                    # print(res.get())
                else:
                    plot_prediction(dir + gfile,pathC,epsg,plotPredition,supr_resolution,VIIRS_dir)
    if(parallel):
        pool.close()
        pool.join()