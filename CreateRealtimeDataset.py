"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime


from GlobalValues import RAD, RealTimeIncoming_files, validate_with_radar
from GoesProcessing import GoesProcessing
from PlotGoesInput import GOES_visual_verification
from RadarProcessing import RadarProcessing
from SiteInfo import SiteInfo
from VIIRSProcessing import VIIRSProcessing


def create_dummy_uniquetime(hour_frequency=1,minute_frequency=10):
    return [str(h).zfill(2) + str(m).zfill(2) for h in range(0, 24, hour_frequency) for m in range(0, 60, minute_frequency) ]
    # unique_time = []
    # for h in range(0, 24, 1):
    #     for m in range(0, 60, 5):
    #         unique_time.append(str(h).zfill(2) + str(m).zfill(2))
    # return unique_time


def create_realtime_dataset(location, product, verify=False):
    site = SiteInfo(location)
    start_time, end_time = site.start_time, site.end_time
    time_dif = end_time - start_time
    log_path = 'logs/failures_' + location + '_' + str(site.start_time) + '_' + str(site.end_time) + '.txt'
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    goes_tif_dir = RealTimeIncoming_files+"/"+location+"/"
    # initialize Goes object and prvide file for log
    goes = GoesProcessing(log_path,list(map(lambda item: item['product_name'], product)),list(map(lambda item: item['band'],product))
                                    ,site=site)
    # initialize VIIRS object , this will create firefixel for particular site and define image parameters
    v2r_viirs = VIIRSProcessing(year=str(start_time.year), satellite="viirs-snpp", site=site)
    radarprocessing = RadarProcessing(location)

    # running for each date
    for i in range(time_dif.days):
        fire_date = str(start_time + datetime.timedelta(days=i))
        if(validate_with_radar):
            # file = f'radar_data/{location}'
            # unique_time = radar_dates(fire_date.replace('-', ''), file)
            unique_time = radarprocessing.get_unique_dateTime(fire_date.replace('-', ''))
        else:
            unique_time = create_dummy_uniquetime()
        # running for ever hhmm for perticular date
        for ac_time in unique_time:
            paths = goes.download_goes(fire_date, str(ac_time))
            if -1 not in paths:
                #TODO: ground truth adaptive ,think about it
                goes.nc2tiff(fire_date, ac_time, paths, site, v2r_viirs.image_size, goes_tif_dir)
                if verify:
                    GOES_visual_verification(ac_time, fire_date, paths, site, save=False)
