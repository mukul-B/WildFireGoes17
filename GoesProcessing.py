"""
This script contains GoesProcessing class contain functionality to
to download GOES data for date , product band and mode
and resample it to tif file for particular site

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import datetime as dt

import numpy as np
import s3fs
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
from satpy import Scene

from GlobalValues import RAD, FDC


class GoesProcessing:
    def __init__(self, log_path):
        # failure log ex missing files logged in file
        self.product_name = None
        self.band = None
        self.g_reader = None
        self.failures = open(log_path, 'w')

    def __del__(self):
        self.failures.close()

    # download GOES data for date , product band and mode , finction have default values
    def download_goes(self, fire_date, ac_time, directory, product_name=RAD, band=7, mode='M6',
                      bucket_name='noaa-goes17'):

        # product to g_reader for Satpy
        g_reader = product_name.split('-')
        self.g_reader = '_'.join(g_reader[:2]).lower()
        self.g_reader = 'abi_l2_nc' if (self.g_reader == 'abi_l2') else self.g_reader
        self.band = band
        self.product_name = product_name

        # extract date parameter of AWS request from given date and time
        sDATE = dt.datetime.strptime(fire_date + "_" + ac_time.zfill(4), '%Y-%m-%d_%H%M')
        day_of_year = sDATE.timetuple().tm_yday
        year = sDATE.year
        hour = sDATE.hour
        minute = sDATE.minute
        print(day_of_year, year, hour, minute)

        # Use anonymous credentials to access public data  from AWS
        fs = s3fs.S3FileSystem(anon=True)
        band = ('C' + str(band).zfill(2) if band else "")
        # fdc does have commutative bands
        band = "" if (self.product_name == FDC) else band
        # Write prefix for the files of inteest, and list all files beginning with this prefix.
        prefix = f'{bucket_name}/{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/'
        file_prefix = f'OR_{product_name}-{mode}{band}_G17_s{year}{day_of_year:03.0f}{hour:02.0f}'

        # listing all files for product date for perticular hour
        files = fs.ls(prefix)

        # filtering band and mode from file list
        files = [i for i in files if file_prefix in i]

        # return if file not present for criteria and write in log
        if len(files) == 0:
            print("fine not found in GOES")
            self.failures.write("No Match found for {}\n".format(sDATE))
            return -1

        # find closed goes fire from viirs( the closest minutes)
        last, closest = 0, 0
        for index, file in enumerate(files):
            fname = file.split("/")[-1]
            splits = fname.split("_")
            g_time = dt.datetime.strptime(splits[3], 's%Y%j%H%M%S%f')
            if int(dt.datetime.strftime(g_time, '%M')) < int(minute):
                last = index
            if abs(int(dt.datetime.strftime(g_time, '%M')) - int(minute)) < 3:
                closest = index

        # downloading closed file
        first_file = files[closest]
        # out_file=  "GOES-"+str(fire_date)+"_"+str(ac_time)+".nc"
        out_file = first_file.split('/')[-1]
        path = directory + '/' + product_name + "/" + out_file
        print('\nDownloading files from AWS...', end='', flush=True)
        fs.download(first_file, path)
        print(files[closest], "completed")

        return path

    #   resampling GOES file for given site and writing in a tiff file
    def nc2tiff(self, fire_date, ac_time, path, site, image_size):

        # creating bouding box from site information
        band = self.band
        band = ('C' + str(band).zfill(2) if band else "")
        layer = "Mask" if (self.product_name == FDC) else band
        latitude, longitude = site.latitude, site.longitude
        rectangular_size = site.rectangular_size
        EPSG = site.EPSG

        area_def = self.get_areaDefination(EPSG, image_size, latitude, longitude, rectangular_size)

        # using satpy to crop goes for the given site
        # g_reader = 'abi_l1b_nc'

        # ABI - L1b - RadC
        goes_scene = Scene(reader=self.g_reader,
                           filenames=[path])
        goes_scene.load([layer])
        # print(area_def)
        goes_scene = goes_scene.resample(area_def)
        # print('layer',layer)

        # # trying to filter goes data
        # # print(goes_scene[band])
        x = goes_scene.to_xarray_dataset()
        # rad = x[layer].values
        # # rad[rad < 30] = 0
        # # rad[rad > 35] = 0
        # # print(rad)
        # x[layer].values = rad
        # goes_scene[layer].values = x[layer].values

        # saving output file
        out_file = "GOES-" + str(fire_date) + "_" + str(ac_time)
        out_path = "/".join(path.split('/')[:-1]) + "/tif/" + out_file + '.tif'
        print(out_path)
        # goes_scene[layer].rio.to_raster(raster_path=out_path, driver='GTiff', dtype='float32')

        goes_scene.save_dataset(layer, filename=out_path)
            # , writer='geotiff',dtype=np.float32

    # get area defination for satpy, with new projection and bounding pox
    def get_areaDefination(self, EPSG, image_size, latitude, longitude, rectangular_size):
        bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
        top_right = [latitude + rectangular_size, longitude + rectangular_size]
        # transformining bounding box coordinates for new projection
        transformer = Transformer.from_crs(4326, EPSG)
        bottom_left_utm = [int(transformer.transform(bottom_left[0], bottom_left[1])[0]),
                           int(transformer.transform(bottom_left[0], bottom_left[1])[1])]
        top_right_utm = [int(transformer.transform(top_right[0], top_right[1])[0]),
                         int(transformer.transform(top_right[0], top_right[1])[1])]

        # defining area definition with image size , projection and extend
        area_id = 'given'
        description = 'given'
        proj_id = 'given'
        projection = 'EPSG:' + str(EPSG)
        width = image_size[1]
        height = image_size[0]
        # the lat lon is changed when using utm !?
        area_extent = (bottom_left_utm[0], bottom_left_utm[1], top_right_utm[0], top_right_utm[1])
        area_def = AreaDefinition(area_id, description, proj_id, projection,
                                  width, height, area_extent)
        return area_def
