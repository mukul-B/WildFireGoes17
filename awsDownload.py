import datetime as dt
import os

import pandas as pd
from boto3.session import Session
from netCDF4 import Dataset


def download_goes(date, hour, min, satname='goes17', product='ABI-L2-FDCC', mode='M6'):

    # creating date range
    nexth = str((int(hour) + 1) % 24)
    sDATE = dt.datetime.strptime(date + "_" + hour, '%Y-%m-%d_%H')  # Start/end time (YYYY, M, d, H)
    eDATE = dt.datetime.strptime(date + "_" + nexth, '%Y-%m-%d_%H')

    if nexth == "0":
        eDATE = eDATE + dt.timedelta(days=1)
    date_list = pd.date_range(sDATE, eDATE, freq='1H')[:-1]

    # Aws connection
    data = pd.read_csv("awsKey.csv")
    ACCESS_KEY = data["ACCESS_KEY"][0]
    SECRET_KEY = data["SECRET_KEY"][0]

    session = Session(aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    s3 = session.client('s3')

    # download directory
    keyword = 'goestemp'
    mainpath = '' + keyword
    filepath = mainpath + '/files'
    filename = mainpath + '/files/satfile2.nc'

    if not os.path.exists(mainpath):
        os.makedirs(mainpath)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for i in range(len(date_list)):
        bucket = 'noaa-' + satname
        prefix = product + '/' + str(date_list[i].year) \
                 + '/' + str(date_list[i].day_of_year).zfill(3) \
                 + '/' + str(date_list[i].hour).zfill(2) \
                 + '/OR_' + product + '-' + mode
        # print(bucket,prefix)
        filelist = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        # print(filelist)
        filelist=filelist['Contents']

        for key in range(len(filelist)):
            s3.download_file(bucket, filelist[key]['Key'], filename)
            data = Dataset(filename, 'r')
            midpoint = float(data.variables['t'][:])
            scan_mid = dt.datetime(2000, 1, 1, 12) + dt.timedelta(seconds=midpoint)
            # projInfo = data.variables['goes_imager_projection']
            # print(dt.datetime.strftime(scan_mid, '%M'))
            if abs(int(dt.datetime.strftime(scan_mid, '%M')) - int(min)) <3:
                print('\nDownloading files from AWS...', end='', flush=True)
                print(dt.datetime.strftime(scan_mid, '%Y%m%d_%H%M'))
                break
            data.close()
            data = None
