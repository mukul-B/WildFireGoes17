import datetime as dt
import os
import pandas as pd
from boto3.session import Session
from netCDF4 import Dataset


def download_goes(date, hour, min):
    nexth = str(int(hour) + 1)
    sDATE = dt.datetime.strptime(date + "_" + hour, '%Y-%m-%d_%H')  # Start/end time (YYYY, M, d, H)
    eDATE = dt.datetime.strptime(date + "_" + nexth, '%Y-%m-%d_%H')
    satname = 'goes17'  # goes16 or goes17
    product = 'ABI-L2-FDCC'  # Fire product
    mode = 'M6'  # usually M6 but M3 for older data?
    keyword = 'goestemp'
    mainpath = '' + keyword
    filepath = mainpath + '/files'
    data = pd.read_csv("awsKey.csv")
    ACCESS_KEY = data["ACCESS_KEY"][0]
    SECRET_KEY = data["SECRET_KEY"][0]

    if not os.path.exists(mainpath):
        os.makedirs(mainpath)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    date_list = pd.date_range(sDATE, eDATE, freq='1H')[:-1]

    session = Session(aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    s3 = session.client('s3')

    for i in range(len(date_list)):

        bucket = 'noaa-' + satname
        prefix = product + '/' + str(date_list[i].year) + '/' + str(date_list[i].day_of_year).zfill(3) + '/' + str(
            date_list[i].hour).zfill(2) + '/OR_' + product + '-' + mode
        filelist = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']

        for key in range(len(filelist)):

            s3.download_file(bucket, filelist[key]['Key'], mainpath + '/files/satfile2.nc')
            filename = mainpath + '/files/satfile2.nc'
            data = Dataset(filename, 'r')
            midpoint = float(data.variables['t'][:])
            scan_mid = dt.datetime(2000, 1, 1, 12) + dt.timedelta(seconds=midpoint)
            projInfo = data.variables['goes_imager_projection']
            if (dt.datetime.strftime(scan_mid, '%M') == min):
                print(filelist[key])
                print('\nDownloading files from AWS...', end='', flush=True)
                print(dt.datetime.strftime(scan_mid, '%Y%m%d_%H%M'))

                break
            data.close()
            data = None
