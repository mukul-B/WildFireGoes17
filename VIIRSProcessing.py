"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import numpy
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

from GlobalValues import viirs_dir, VIIRS_OVERWRITE
from os.path import exists as file_exists


class VIIRSProcessing:
    def __init__(self, year="2021", satellite="viirs-snpp", site=None, res=375):

        country = 'United_States'
        Sdirectory = "VIIRS_Source/" + satellite + "_" + year + "_" + country + ".csv"
        VIIRS_pixel= pd.read_csv(Sdirectory)
        # Sdirectory3 = f'VIIRS_Source_new/fire_archive_SV-C2_{year}.csv'
        # Sdirectory2 = f'VIIRS_Source_new/fire_nrt_J1V-C2_{year}.csv'
        # snpp_pixels = pd.read_csv(Sdirectory3)
        # NOAA_pixels = pd.read_csv(Sdirectory2)
        # NOAA_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
        # NOAA_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
        # VIIRS_pixel = pd.concat([snpp_pixels, NOAA_pixels], ignore_index=True)
        # VIIRS_pixel = NOAA_pixels

        self.location = site.location
        self.fire_pixels = VIIRS_pixel
        self.satellite = satellite
        self.crs = site.EPSG
        self.res = res

        # defining extend of site in lat and lon
        latitude, longitude = site.latitude, site.longitude
        rectangular_size = site.rectangular_size
        bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
        top_right = [latitude + rectangular_size, longitude + rectangular_size]
        # print(bottom_left, top_right)

        # filtering in fire pixel inside the bounding box of the given site
        self.fire_pixels = self.fire_pixels[self.fire_pixels.latitude.gt(bottom_left[0])
                                            & self.fire_pixels.latitude.lt(top_right[0])
                                            & self.fire_pixels.longitude.gt(bottom_left[1])
                                            & self.fire_pixels.longitude.lt(top_right[1])]

        # transforming lon lat to utm
        # UTM, Universal Transverse Mercator ( northing and easting)
        # https://www.youtube.com/watch?v=LcVlx4Gur7I

        # https://epsg.io/32611
        # 32610 (126 to 120) ;32611 (120 to 114) ;32612 (114 to 108)

        self.transformer = Transformer.from_crs(4326, self.crs)
        bottom_left_utm = [int(self.transformer.transform(bottom_left[0], bottom_left[1])[0]),
                           int(self.transformer.transform(bottom_left[0], bottom_left[1])[1])]
        top_right_utm = [int(self.transformer.transform(top_right[0], top_right[1])[0]),
                         int(self.transformer.transform(top_right[0], top_right[1])[1])]

        top_right_utm = [top_right_utm[0] - (top_right_utm[0] - bottom_left_utm[0]) % self.res,
                         top_right_utm[1] - (top_right_utm[1] - bottom_left_utm[1]) % self.res]
        # adjustment (adding residue) because we want to make equal sized grids on whole area
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
        self.nx = round((self.xmax - self.xmin) / self.res)
        self.ny = round((self.ymax - self.ymin) / self.res)
        self.image_size = (self.ny, self.nx)

    def make_tiff(self,  fire_date,ac_time, fire_data_filter_on_date_and_bbox):

        # output file name
        viirs_tif_dir = viirs_dir.replace('$LOC', self.location)
        out_file = viirs_tif_dir + self.satellite + '-' + str(fire_date) + "_" + str(ac_time) + '.tif'

        if ((not VIIRS_OVERWRITE) and file_exists(out_file)):
            return
        # filter firepixel for time of date
        fire_data_filter_on_time = fire_data_filter_on_date_and_bbox[
            fire_data_filter_on_date_and_bbox.acq_time.eq(ac_time)]
        fire_data_filter_on_timestamp = np.array(fire_data_filter_on_time)

        # b1_pixels = self.inverse_mapping(fire_data_filter_on_timestamp)

        # exit(0)
        # creating pixel values used in tiff
        b1_pixels,b2_pixels = self.create_raster_array(fire_data_filter_on_timestamp)
        b1_pixels = self.interpolation(b1_pixels)
        # comment normalize_VIIRS for visualizing real values
        # b1_pixels = self.normalize_VIIRS(b1_pixels)
        # print("--------------------", np.max(b1_pixels))
        self.gdal_writter(out_file,[b1_pixels,b2_pixels])

    def normalize_VIIRS(self, b1_pixels):
        max_val = np.max(b1_pixels)
        # max_val = 367
        if max_val > 1:
            b1_pixels = (b1_pixels / max_val) * 255
        b1_pixels = b1_pixels.astype(int)
        return b1_pixels

    def inverse_mapping(self, fire_data_filter_on_timestamp):
        b2_pixels = np.zeros(self.image_size, dtype=float)
        neigh = NearestNeighbors(n_neighbors=4, radius=375)
        coords = []
        radiance = []
        for k in range(fire_data_filter_on_timestamp.shape[0]):
            record = fire_data_filter_on_timestamp[k]
            # transforming lon lat to utm
            lon_point = self.transformer.transform(record[0], record[1])[0]
            lat_point = self.transformer.transform(record[0], record[1])[1]
            coords.append([lat_point, lon_point])
            radiance.append(record[2])
        if not coords:
            return b2_pixels
        neigh.fit(coords)

        for ii in range(b2_pixels.shape[1]):
            for jj in range(b2_pixels.shape[0]):
                lon = ii * self.res + self.xmin
                lat = jj * self.res + self.ymin
                d = neigh.radius_neighbors([[lat, lon]], 266, return_distance=True)
                index_list = d[1][0]
                rad2 = 0
                n2 = 0
                for k, ind in enumerate(index_list):
                    rad2 += radiance[ind]
                    n2 += 1
                if n2 > 0:
                    b2_pixels[-jj, ii] = rad2 / n2
        return b2_pixels


    def inverse_mapping2(self, fire_data_filter_on_timestamp):
        hm = {}
        b1_pixels = np.zeros(self.image_size, dtype=float)
        b2_pixels = np.zeros(self.image_size, dtype=float)

        neigh = NearestNeighbors(n_neighbors=4, radius=375)
        coords = []
        radiance = []
        for k in range(1, fire_data_filter_on_timestamp.shape[0]):
            record = fire_data_filter_on_timestamp[k]
            # transforming lon lat to utm
            lon_point = self.transformer.transform(record[0], record[1])[0]
            lat_point = self.transformer.transform(record[0], record[1])[1]
            cord_x = round((lon_point - self.xmin) / self.res)
            cord_y = round((lat_point - self.ymin) / self.res)
            # if cord_x >= self.nx or cord_y >= self.ny:
            #     continue
            coords.append([lat_point, lon_point])
            radiance.append(record[2])
            # b1_pixels[-cord_y, cord_x] = max(b1_pixels[-cord_y, cord_x], record[2])

        if not coords:
            return b2_pixels
        neigh.fit(coords)

        # for ii in range(b2_pixels.shape[1]):
        #     for jj in range(b2_pixels.shape[0]):
        #         lon = ii * self.res + self.xmin
        #         lat = jj * self.res + self.ymin
        #         d = neigh.radius_neighbors([[lat, lon]], 266, return_distance=True)
        #         index_list = d[1][0]
        #         rad2 = 0
        #         n2 = 0
        #         for k, ind in enumerate(index_list):
        #             rad2 += radiance[ind]
        #             n2 += 1
        #         if n2 > 0:
        #             b2_pixels[-jj, ii] = rad2 / n2
        # return b2_pixels
        # fig, axs = plt.subplots(2, 3, constrained_layout=True)
        # axs[0][0].imshow(b1_pixels)
        # axs[0][0].set_title("old approach")
        lp = 0
        for dt in [0]:
            lp += 1
            for ii in range(b2_pixels.shape[1]):
                for jj in range(b2_pixels.shape[0]):
                    lon = ii * self.res + self.xmin
                    lat = jj * self.res + self.ymin
                    # if (jj, ii) in hm:
                    #     fg = hm[jj, ii]
                    #     lat2, lon2, rad2 = fg[-1]
                    #     b1_pixels[-jj, ii] = max(b1_pixels[-jj, ii], rad2)
                    if True:
                        d = neigh.kneighbors([[lat, lon]], min(len(coords),16), return_distance=True)
                        # d = neigh.radius_neighbors([[lat, lon]], 266, return_distance=True)
                        index_list = d[1][0]
                        distance_list = d[0][0]
                        rad2 = 0
                        n2 = 0
                        n3 = 0
                        x = numpy.array([])
                        y = numpy.array([])
                        z = numpy.array([])
                        for k, ind in enumerate(index_list):
                            # 187.5
                            # 265.165
                            if  distance_list[k] <= 266:
                                rad2 += radiance[ind]
                                n2 += 1
                            if distance_list[k] < dt:
                                x = np.append(x,[coords[ind][0]])
                                y = np.append(y,[coords[ind][1]])
                                z = np.append(z,[radiance[ind]])
                                n3 += 1

                        if n2 > 0:
                            b1_pixels[-jj, ii] = rad2 / n2

                        if  n3 >= 16:
                            # pass
                            # print(n3)
                            f = interpolate.interp2d(x, y, z, kind='cubic')
                            try:
                                v = f(lat, lon)
                                if v < 400 and v > 200:
                                    b2_pixels[-jj, ii] = v
                            except:
                                print("error")
                                continue
        kol = np.zeros(self.image_size, dtype=float)
        for ii in range(b2_pixels.shape[1]):
            for jj in range(b2_pixels.shape[0]):
                if b1_pixels[-jj][ii]  > 0 and b2_pixels[-jj][ii]  > 0:
                    # print(b2_pixels[-jj][ii] - b1_pixels[-jj][ii])
                    kol[-jj][ii]  = b2_pixels[-jj][ii] - b1_pixels[-jj][ii]

        return b2_pixels

    def create_raster_array(self, fire_data_filter_on_timestamp):
        b1_pixels = np.zeros(self.image_size, dtype=float)
        b2_pixels = np.zeros(self.image_size, dtype=float)
        for k in range(1, fire_data_filter_on_timestamp.shape[0]):
            record = fire_data_filter_on_timestamp[k]
            # transforming lon lat to utm
            lon_point = self.transformer.transform(record[0], record[1])[0]
            lat_point = self.transformer.transform(record[0], record[1])[1]
            cord_x = round((lon_point - self.xmin) / self.res)
            cord_y = round((lat_point - self.ymin) / self.res)
            if cord_x >= self.nx or cord_y >= self.ny:
                continue
            # print(k, -cord_y, cord_x, fire_date, ac_time, (lon_point - self.xmin) , (lat_point - self.ymin) )
            # 122 -140 58 2021-07-16 1012 21926.29492325522 52647.5308539886
            # 137 -140 58 2021-07-16 1012 21806.47712273756 52779.97441741172
            # writing bright_ti4 ( record[2] )to tif
            modified_BT = record[2]
            if(record[2] ==208):
                modified_BT = 367
            if(record[2]<record[11] and record[11]<=367):
                modified_BT = record[11]
            # if(modified_BT > 367):
            #     modified_BT = 367
            
            
            b1_pixels[-cord_y, cord_x] = max(b1_pixels[-cord_y, cord_x], modified_BT)
            b2_pixels[-cord_y, cord_x] = max(b2_pixels[-cord_y, cord_x], record[12])
        return b1_pixels,b2_pixels

    # check if the zero is farbackground or surronding the fire, used for interpolation
    def nonback_zero(self, b1_pixels, ii, jj):
        checks = [
            (ii - 1, jj - 1), (ii, jj - 1), (ii + 1, jj - 1),
            (ii - 1, jj), (ii + 1, jj),
            (ii - 1, jj + 1), (ii, jj + 1), (ii + 1, jj + 1)
        ]
        for m, n in checks:
            if b1_pixels[m, n] != 0.0:
                return True
        return False

    def interpolation(self, b1_pixels):
        grid_x = np.linspace(self.xmin, self.xmax, self.nx)
        grid_y = np.linspace(self.ymin, self.ymax, self.ny)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        filtered_b12 = []
        bia_x2 = []
        bia_y2 = []
        chec = []
        for ii in range(b1_pixels.shape[0]):
            for jj in range(b1_pixels.shape[1]):
                if ii != 0 and ii != (b1_pixels.shape[0] - 1) and jj != 0 and jj != (b1_pixels.shape[1] - 1):
                    # skiping the border values
                    if b1_pixels[ii, jj] == 0:
                        # interpolation zeros which have at least one surrounding fire pixel
                        if self.nonback_zero(b1_pixels, ii, jj):
                            chec.append((ii, jj))
                            continue
                filtered_b12.append(b1_pixels[ii, jj])
                bia_x2.append(grid_x[ii, jj])
                bia_y2.append(grid_y[ii, jj])
        filtered_b1 = np.array(filtered_b12)
        grid_xx = np.array(bia_x2).reshape(-1, 1)
        grid_yy = np.array(bia_y2).reshape(-1, 1)
        grid = np.hstack((grid_xx, grid_yy))
        grid_z = griddata(grid, filtered_b1, (grid_x, grid_y), method='nearest', fill_value=0)
        # grid_z = griddata(grid, filtered_b1, (grid_x, grid_y), method='linear', fill_value=0)
        # grid_z = griddata(grid, filtered_b1, (grid_x, grid_y), method='cubic', fill_value=0)
        # plot_sample([b1_pixels, grid_z,grid_z_l,grid_z_c], ["Rasterized VIIRS", "nearest","linear","cubic"])

        return grid_z

    def gdal_writter(self, out_file, b_pixels):
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, self.image_size[1],
            self.image_size[0], len(b_pixels),
            gdal.GDT_Float32)
        # transforms between pixel raster space to projection coordinate space.
        # new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        geotransform = (self.xmin, self.res, 0, self.ymax, 0, -self.res)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(self.crs)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        for i in range(len(b_pixels)):
            dst_ds.GetRasterBand(i+1).WriteArray(b_pixels[i])
        # dst_ds.GetRasterBand(1).WriteArray(b1_pixels)  # write r-band to the raster
        # dst_ds.GetRasterBand(2).WriteArray(b2_pixels)
        dst_ds.FlushCache()  # write to disk
        dst_ds = None
