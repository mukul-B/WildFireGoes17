"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime

from GlobalValues import RAD
from GoesProcessing import GoesProcessing
from SiteInfo import SiteInfo
from VIIRSProcessing import VIIRSProcessing


def createDataset(location, product_name=RAD):
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

        # filter firepixel for a date
        fire_data_filter_on_date_and_bbox = v2r_viirs.fire_pixels[v2r_viirs.fire_pixels.acq_date.eq(fire_date)]
        unique_time = fire_data_filter_on_date_and_bbox.acq_time.unique()

        # running for ever hhmm for perticular date
        for ac_time in unique_time[:1]:
            print(fire_date, ac_time)
            path = goes.download_goes(fire_date, str(ac_time), 'data/' + location + '/GOES', product_name=product_name)

            if (path != -1):
                v2r_viirs.make_tiff(ac_time, fire_date, fire_data_filter_on_date_and_bbox)
                goes.nc2tiff(fire_date, ac_time, path, site,v2r_viirs.image_size)

