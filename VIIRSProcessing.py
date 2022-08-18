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

from SiteInfo import SiteInfo


class VIIRSProcessing:
    def __init__(self, year="2021", satellite="viirs-snpp", site=SiteInfo('dixie'), crs=32611, res=375):

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

        lon = [bottom_left_utm[0], top_right_utm[0]]
        lat = [bottom_left_utm[1], top_right_utm[1]]
        # print(lon, lat)

        # setting image parameters
        self.xmin, self.ymin, self.xmax, self.ymax = [min(lon), min(lat), max(lon), max(lat)]
        self.nx = int((self.xmax - self.xmin) // self.res)
        self.ny = int((self.ymax - self.ymin) // self.res)
        self.image_size = (self.ny, self.nx)

        # self.transformer2 = Transformer.from_crs(4326, self.crs)

    def make_tiff(self, ac_time, fire_date, fire_data_filter_on_date_and_bbox):

        # filter firepixel for time of date
        fire_data_filter_on_time = fire_data_filter_on_date_and_bbox[
            fire_data_filter_on_date_and_bbox.acq_time.eq(ac_time)]
        fire_data_filter_on_timestamp = np.array(fire_data_filter_on_time)

        # creating pixel values used in tiff
        b1_pixels = np.zeros(self.image_size, dtype=float)
        for k in range(1, fire_data_filter_on_timestamp.shape[0]):
            record = fire_data_filter_on_timestamp[k]

            # transforming lon lat to utm
            lon_point = self.transformer.transform(record[0], record[1])[0]
            lat_point = self.transformer.transform(record[0], record[1])[1]

            cord_x = int((lon_point - self.xmin) // self.res)
            cord_y = int((lat_point - self.ymin) // self.res)
            if cord_x >= self.nx or cord_y >= self.ny:
                continue
            # writing bright_ti4 ( record[2] )to tif
            b1_pixels[-cord_y, cord_x] = max(b1_pixels[-cord_y, cord_x], record[2])

        out_file = 'data/' + self.location + '/VIIRS/' + 'FIRMS' + '-' + \
                   str(fire_date) + "_" + str(ac_time) + '.tif'

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
        print(b1_pixels.shape)
        dst_ds.FlushCache()  # write to disk
        dst_ds = None
