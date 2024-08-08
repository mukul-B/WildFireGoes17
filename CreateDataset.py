"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime

from GlobalValues import RAD, goes_dir
from GoesProcessing import GoesProcessing
from SiteInfo import SiteInfo
from VIIRSProcessing import VIIRSProcessing


def createDataset(location, product):
#     product_name = product['product_name']
#     band = product['band']
    site = SiteInfo(location)
    start_time, end_time = site.start_time, site.end_time
    time_dif = end_time - start_time
    log_path = 'logs/failures_' + location + '_' + str(site.start_time) + '_' + str(site.end_time) + '.txt'
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product['product_name']).replace('$BAND', format(product['band'],'02d'))
                
    # initialize Goes object and prvide file for log
    goes = GoesProcessing(log_path,list(map(lambda item: item['product_name'], product)),list(map(lambda item: item['band'], product)))
    # initialize VIIRS object , this will create firefixel for particular site and define image parameters
    v2r_viirs = VIIRSProcessing(year=str(start_time.year), satellite="viirs-snpp", site=site)
    v2r_viirs.extract_hotspots()

    # running for each date
    for i in range(time_dif.days):
        fire_date = str(start_time + datetime.timedelta(days=i))

        # filter firepixel for a date
        fire_data_filter_on_date_and_bbox = v2r_viirs.fire_pixels[v2r_viirs.fire_pixels.acq_date.eq(fire_date)]
        unique_time = fire_data_filter_on_date_and_bbox.acq_time.unique()
        # running for ever hhmm for perticular date
        for ac_time in unique_time:
            # print(fire_date, ac_time)
            # path = goes.download_goes(fire_date, str(ac_time), product_name=product_name,band=band) 
            paths = goes.download_goes(fire_date, str(ac_time))
            if -1 not in paths:
                v2r_viirs.make_tiff(fire_date, ac_time, fire_data_filter_on_date_and_bbox)
                goes.nc2tiff(fire_date, ac_time, paths, site, v2r_viirs.image_size, goes_tif_dir)

