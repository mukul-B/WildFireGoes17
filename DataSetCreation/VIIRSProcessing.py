"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer

from GlobalValues import viirs_dir
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


class VIIRSProcessing:
    def __init__(self, year="2021", satellite="viirs-snpp", site=None, res=375):

        country = 'United_States'
        Sdirectory = "VIIRS_Source/" + satellite + "_" + year + "_" + country + ".csv"

        self.location = site.location
        self.fire_pixels = pd.read_csv(Sdirectory)
        self.crs = site.EPSG
        self.res = res

        # defining extend of site in lat and lon
        latitude, longitude = site.latitude, site.longitude
        rectangular_size = site.rectangular_size
        bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
        top_right = [latitude + rectangular_size, longitude + rectangular_size]
        print(bottom_left, top_right)

        # filtering in fire pixel inside the bounding box of the given site
        self.fire_pixels = self.fire_pixels[self.fire_pixels.latitude.gt(bottom_left[0])
                                            & self.fire_pixels.latitude.lt(top_right[0])
                                            & self.fire_pixels.longitude.gt(bottom_left[1])
                                            & self.fire_pixels.longitude.lt(top_right[1])]

        # transforming lon lat to utm
        # UTM, Universal Transverse Mercator ( northing and easting)
        # https://www.youtube.com/watch?v=LcVlx4Gur7I

        # https://epsg.io/32611

        self.transformer = Transformer.from_crs(4326, self.crs)
        bottom_left_utm = [int(self.transformer.transform(bottom_left[0], bottom_left[1])[0]),
                           int(self.transformer.transform(bottom_left[0], bottom_left[1])[1])]
        top_right_utm = [int(self.transformer.transform(top_right[0], top_right[1])[0]),
                         int(self.transformer.transform(top_right[0], top_right[1])[1])]

        top_right_utm = [top_right_utm[0] - (top_right_utm[0] - bottom_left_utm[0]) % self.res,
                         top_right_utm[1] - (top_right_utm[1] - bottom_left_utm[1]) % self.res]
        # ------
        # ------
        # ------
        # ------
        # creating offset for top right pixel

        lon = [bottom_left_utm[0], top_right_utm[0]]
        lat = [bottom_left_utm[1], top_right_utm[1]]
        # print(lon, lat)

        # setting image parameters
        self.xmin, self.ymin, self.xmax, self.ymax = [min(lon), min(lat), max(lon), max(lat)]
        self.nx = int((self.xmax - self.xmin) // self.res)
        self.ny = int((self.ymax - self.ymin) // self.res)
        self.image_size = (self.ny, self.nx)

    def make_tiff(self, ac_time, fire_date, fire_data_filter_on_date_and_bbox):

        # output file name
        viirs_tif_dir = viirs_dir.replace('$LOC', self.location)
        out_file = viirs_tif_dir + 'FIRMS' + '-' + str(fire_date) + "_" + str(ac_time) + '.tif'

        # filter firepixel for time of date
        fire_data_filter_on_time = fire_data_filter_on_date_and_bbox[
            fire_data_filter_on_date_and_bbox.acq_time.eq(ac_time)]
        fire_data_filter_on_timestamp = np.array(fire_data_filter_on_time)

        # creating pixel values used in tiff
        b1_pixels = np.zeros(self.image_size, dtype=float)
        x, y = [], []
        value = []
        for k in range(1, fire_data_filter_on_timestamp.shape[0]):
            record = fire_data_filter_on_timestamp[k]

            # transforming lon lat to utm
            lon_point = self.transformer.transform(record[0], record[1])[0]
            lat_point = self.transformer.transform(record[0], record[1])[1]

            cord_x = int((lon_point - self.xmin) // self.res)
            cord_y = int((lat_point - self.ymin) // self.res)
            if cord_x >= self.nx or cord_y >= self.ny:
                continue
            # print(k, -cord_y, cord_x, fire_date, ac_time, (lon_point - self.xmin) , (lat_point - self.ymin) )
            # 122 -140 58 2021-07-16 1012 21926.29492325522 52647.5308539886
            # 137 -140 58 2021-07-16 1012 21806.47712273756 52779.97441741172
            # writing bright_ti4 ( record[2] )to tif
            x.append(lon_point)
            y.append(lat_point)

            # b1_pixels[-cord_y, cord_x] = max(b1_pixels[-cord_y, cord_x], record[2])
            b1_pixels[-cord_y, cord_x] = record[2]
            value.append(record[2])

        # b1_pixels = self.interpolation(b1_pixels, value, x, y)

        self.gdal_writter(b1_pixels, out_file)

    def interpolation(self, b1_pixels, value, x, y):
        #     --------------------
        #     ------.-------------
        #     --------------------
        # for i in range(len(b1_pixels)):
        #     for j in range(len(b1_pixels[0])):
        #         x.append(i)
        #         y.append(j)
        #         value.append(b1_pixels[i, j])
        points = np.array([x, y]).T
        values = np.array(value)
        print(points)
        print(values)
        print(self.xmin, self.xmax, self.nx, int((self.xmax - self.xmin) // self.res))
        print("------------")
        print(self.ymin, self.ymax, self.ny)
        grid_x = np.linspace(self.xmin, self.xmax, self.nx)
        grid_y = np.linspace(self.ymin, self.ymax, self.ny)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_xx = grid_x.flatten().reshape(-1, 1)
        grid_yy = grid_y.flatten().reshape(-1, 1)
        grid = np.hstack((grid_xx, grid_yy))
        # grid_z = griddata(points, values, (grid_x,grid_y), method='linear',fill_value=0)
        grid_z = griddata(grid, b1_pixels.flatten(), (grid_x, grid_y), method='cubic', fill_value=0)
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        axs[0].imshow(grid_z)
        # cubic
        axs[1].imshow(b1_pixels)
        axs[2].imshow(grid_z - b1_pixels)
        plt.show()
        exit(0)
        if np.max(b1_pixels) > 1:
            b1_pixels = (b1_pixels / np.max(b1_pixels)) * 255
        print("--------------------", np.max(b1_pixels))
        return b1_pixels

    def gdal_writter(self, b1_pixels, out_file):
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, self.image_size[1],
            self.image_size[0], 1,
            gdal.GDT_Float32)
        # transforms between pixel raster space to projection coordinate space.
        # new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        geotransform = (self.xmin, self.res, 0, self.ymin, 0, self.res)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(self.crs)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        dst_ds.GetRasterBand(1).WriteArray(b1_pixels)  # write r-band to the raster
        dst_ds.FlushCache()  # write to disk
        dst_ds = None