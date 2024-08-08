"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime
import os

import pandas as pd

from GlobalValues import RAD, GOES_product, realtimeSiteList, RealTimeIncoming_files, RealTimeIncoming_results, videos
from GoesProcessing import GoesProcessing
from PlotGoesInput import GOES_visual_verification
from SiteInfo import SiteInfo
from VIIRSProcessing import VIIRSProcessing


def radar_dates(date, file):
    radar_list = os.listdir(file)
    date_list = []
    for v_file in sorted(radar_list):
        if not v_file.startswith('._'):
            try:
                if v_file.split("_")[1][:8] == date:
                    date_list.append(v_file.split("_")[1][-4:])
            except:
                continue

    return date_list


def create_dummy_uniquetime(hour_frequency=1,minute_frequency=60):
    return [str(h).zfill(2) + str(m).zfill(2) for h in range(0, 24, hour_frequency) for m in range(0, 60, minute_frequency) ]
    # unique_time = []
    # for h in range(0, 24, 1):
    #     for m in range(0, 60, 5):
    #         unique_time.append(str(h).zfill(2) + str(m).zfill(2))
    # return unique_time


def create_realtime_dataset(location, product=RAD, verify=False):
    site = SiteInfo(location)
    # print(site,"------------------",location)
    start_time, end_time = site.start_time, site.end_time
    time_dif = end_time - start_time

    log_path = 'logs/failures_' + location + '_' + str(site.start_time) + '_' + str(site.end_time) + '.txt'
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)

    # initialize Goes object and prvide file for log
    # goes = GoesProcessing(log_path)
    goes = GoesProcessing(log_path,list(map(lambda item: item['product_name'], product)),list(map(lambda item: item['band'], product)))
    # initialize VIIRS object , this will create firefixel for particular site and define image parameters
    v2r_viirs = VIIRSProcessing(year=str(start_time.year), satellite="viirs-snpp", site=site)

    # running for each date
    for i in range(time_dif.days):
        fire_date = str(start_time + datetime.timedelta(days=i))
        validation_source = False
        if(validation_source):
            file = f'radar_data/{location}'
            unique_time = radar_dates(fire_date.replace('-', ''), file)
        else:
            unique_time = create_dummy_uniquetime()
        # print(fire_date, unique_time)
        # running for ever hhmm for perticular date
        for ac_time in unique_time:
            print(fire_date, ac_time)
            path = goes.download_goes(fire_date, str(ac_time))
            if path != -1:
                goes.nc2tiff(fire_date, ac_time, path, site, v2r_viirs.image_size, RealTimeIncoming_files+"/"+location+"/")
                if verify:
                    GOES_visual_verification(ac_time, fire_date, path, site, save=False)


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
