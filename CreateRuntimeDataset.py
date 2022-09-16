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

from GoesProcessing import GoesProcessing
from VIIRSProcessing import VIIRSProcessing
from GlobalValues import RAD, runtime_dir, runtimeSiteList, RunTimeIncoming_files, RunTimeIncoming_results, videos
from SiteInfo import SiteInfo


def create_runtime_dataset(location, product_name=RAD):
    site = SiteInfo(location)
    start_time, end_time = site.start_time, site.end_time
    time_dif = end_time - start_time
    log_path = 'logs/failures_' + location + '_' + str(site.start_time) + '_' + str(site.end_time) + '.txt'

    # initialize Goes object and prvide file for log
    goes = GoesProcessing(log_path)
    # initialize VIIRS object , this will create firefixel for particular site and define image parameters
    v2r_viirs = VIIRSProcessing(year=str(start_time.year), satellite="viirs-snpp", site=site)

    # running for each date
    for i in range(time_dif.days):
        fire_date = str(start_time + datetime.timedelta(days=i))
        unique_time = ['0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500',
                       '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
        # running for ever hhmm for perticular date
        for ac_time in unique_time:
            # print(fire_date, ac_time)
            path = goes.download_goes(fire_date, str(ac_time), product_name=product_name)
            if (path != -1):
                goes.nc2tiff(fire_date, ac_time, path, site, v2r_viirs.image_size, runtime_dir)


def prepareDir():
    if not os.path.exists(RunTimeIncoming_files):
        os.mkdir(RunTimeIncoming_files)
    if not os.path.exists(RunTimeIncoming_results):
        os.mkdir(RunTimeIncoming_results)
    if not os.path.exists(videos):
        os.mkdir(videos)


if __name__ == '__main__':
    print(runtimeSiteList)
    data = pd.read_csv(runtimeSiteList)
    locations = data["Sites"]
    product = RAD
    # pipeline run for sites mentioned in toExecuteSiteList
    for location in locations:
        print(location)
        prepareDir()
        create_runtime_dataset(location, product_name=product)
