"""
This script will
create (per site per date) tiff from VIIRS csv
and download corresponding GOES file and crop it for site bounding box and create TIFF

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import os
import numpy
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

from GlobalValues import VIIRS_tiff_file_name, viirs_dir, VIIRS_OVERWRITE
from os.path import exists as file_exists


class VIIRSProcessing:
    # def __init__(self, year="2021", satellite="viirs-snpp", site=None, res=375):


    #     self.location = site.location
    #     self.satellite = satellite
    #     self.crs = site.EPSG
    #     self.res = res
    #     self.year = year

    #     # defining extend of site in lat and lon
    #     latitude, longitude = site.latitude, site.longitude
    #     rectangular_size = site.rectangular_size
    #     self.bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
    #     self.top_right = [latitude + rectangular_size, longitude + rectangular_size]
    #     # print(bottom_left, top_right)

    #     # filtering in fire pixel inside the bounding box of the given site

    #     # transforming lon lat to utm
    #     # UTM, Universal Transverse Mercator ( northing and easting)
    #     # https://www.youtube.com/watch?v=LcVlx4Gur7I

    #     # https://epsg.io/32611
    #     # 32610 (126 to 120) ;32611 (120 to 114) ;32612 (114 to 108)

    #     self.transformer = Transformer.from_crs(4326, self.crs)
    #     bottom_left_utm = [int(self.transformer.transform(self.bottom_left[0], self.bottom_left[1])[0]),
    #                        int(self.transformer.transform(self.bottom_left[0], self.bottom_left[1])[1])]
    #     top_right_utm = [int(self.transformer.transform(self.top_right[0], self.top_right[1])[0]),
    #                      int(self.transformer.transform(self.top_right[0], self.top_right[1])[1])]

    #     top_right_utm = [top_right_utm[0] - (top_right_utm[0] - bottom_left_utm[0]) % self.res,
    #                      top_right_utm[1] - (top_right_utm[1] - bottom_left_utm[1]) % self.res]
    #     # adjustment (adding residue) because we want to make equal sized grids on whole area
    #     # ------
    #     # ------
    #     # ------
    #     # ------
    #     # creating offset for top right pixel

    #     lon = [bottom_left_utm[0], top_right_utm[0]]
    #     lat = [bottom_left_utm[1], top_right_utm[1]]
    #     # print(lon, lat)

    #     # setting image parameters
    #     self.xmin, self.ymin, self.xmax, self.ymax = [min(lon), min(lat), max(lon), max(lat)]
    #     self.nx = round((self.xmax - self.xmin) / self.res)
    #     self.ny = round((self.ymax - self.ymin) / self.res)
    #     self.image_size = (self.ny, self.nx)

    def __init__(self, year="2021", satellite="viirs-snpp", site=None, res=375):


        self.location = site.location
        self.satellite = satellite
        self.crs = site.EPSG
        self.res = res
        self.year = year

        # defining extend of site in lat and lon
        self.bottom_left = site.bottom_left
        self.top_right = site.top_right

        self.transformer = site.transformer
        # self.xmin, self.ymin, self.xmax, self.ymax = [site.transformed_bottom_left[1], site.transformed_bottom_left[0], site.transformed_top_right[1], site.transformed_top_right[0]]
        self.xmin, self.ymin, self.xmax, self.ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        self.nx = site.image_size[1]
        self.ny = site.image_size[0]
        self.image_size = site.image_size
        

    def set_FIRMS_MAP_KEY(self):
        file_path = 'secrets/FIRMS_API_MAP_KEY'
        key_map = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' ')
                key_map[key] = value

        self.MAP_KEY=key_map['realtimekey']



    def extract_hotspots_via_API(self, fire_date):

        filePath = f'VIIRS_Source_realtime/VIIRS_NOAA20_NRT_USA_{fire_date}.csv'
        if os.path.exists(filePath):
            # Read the CSV file into a DataFrame
            VIIRS_pixel = pd.read_csv(filePath)
        else:
            
            sensor_list = ['VIIRS_NOAA20_NRT','VIIRS_SNPP_NRT']
            source_list = []
            for sensor in sensor_list:
                US_url = 'https://firms.modaps.eosdis.nasa.gov/api/country/csv/' + self.MAP_KEY + '/'+sensor+'/USA/1/'+fire_date
                df = pd.read_csv(US_url)
                columns_order = [
                                'latitude', 'longitude', 'bright_ti4', 'scan', 'track', 
                                'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 
                                'version', 'bright_ti5', 'frp', 'daynight'
                            ]

                # Select the columns in the desired order
                df = df[columns_order]
                source_list.append(df)
            VIIRS_pixel = pd.concat(source_list, ignore_index=True)
            VIIRS_pixel.to_csv(filePath, index=False)

        self.fire_pixels = VIIRS_pixel

        self.fire_pixels = self.fire_pixels[self.fire_pixels.latitude.gt(self.bottom_left[0])
                                            & self.fire_pixels.latitude.lt(self.top_right[0])
                                            & self.fire_pixels.longitude.gt(self.bottom_left[1])
                                            & self.fire_pixels.longitude.lt(self.top_right[1])]
    def extract_hotspots(self):
        country = 'United_States'
        dtype = np.dtype([
                ('latitude', 'float64'),
                ('longitude', 'float64'),
                ('bright_ti4', 'float64'),
                ('scan', 'float32'),
                ('track', 'float32'),
                ('acq_date', 'U10'),    # String for date
                ('acq_time', 'int32'),     # Assuming time is stored as an integer
                ('satellite', 'object'),   # String for satellite
                ('instrument', 'object'),  # String for instrument
                ('confidence', 'object'),  # String for confidence
                ('version', 'object'),     # String for version
                ('bright_ti5', 'float64'),
                ('frp', 'float32'),
                ('daynight', 'U1'),    # String for day/night
                ('type', 'int32')          # Integer for type
            ])
        source_list = []
        Sdirectory = "VIIRS_Source/" + self.satellite + "_" + self.year + "_" + country + ".csv"
        if os.path.exists(Sdirectory):
            snn_yearly= pd.read_csv(Sdirectory, low_memory=False)
            source_list.append(snn_yearly)

        Sdirectory3 = f'VIIRS_Source_new/fire_archive_SV-C2_{self.year}.csv'
        if os.path.exists(Sdirectory3):
            snpp_pixels = pd.read_csv(Sdirectory3)
            snpp_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
            snpp_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
            source_list.append(snpp_pixels)

        Sdirectory2 = f'VIIRS_Source_new/fire_nrt_J1V-C2_{self.year}.csv'
        if os.path.exists(Sdirectory2):
            NOAA_pixels = pd.read_csv(Sdirectory2)
            NOAA_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
            NOAA_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
            source_list.append(NOAA_pixels)

        Sdirectory4 = f'VIIRS_Source_new/fire_nrt_SV-C2_{self.year}.csv'
        if os.path.exists(Sdirectory4):
            snpp_nrt_pixels = pd.read_csv(Sdirectory4)
            snpp_nrt_pixels.rename(columns={'brightness':'bright_ti4'}, inplace=True)
            snpp_nrt_pixels.rename(columns={'bright_t31':'bright_ti5'}, inplace=True)
            source_list.append(snpp_nrt_pixels)
         
        VIIRS_pixel = pd.concat(source_list, ignore_index=True)
        # VIIRS_pixel = NOAA_pixels

        self.fire_pixels = VIIRS_pixel

        self.fire_pixels = self.fire_pixels[self.fire_pixels.latitude.gt(self.bottom_left[0])
                                            & self.fire_pixels.latitude.lt(self.top_right[0])
                                            & self.fire_pixels.longitude.gt(self.bottom_left[1])
                                            & self.fire_pixels.longitude.lt(self.top_right[1])]
        
    def get_unique_dateTime(self, fire_date):
        self.fire_data_filter_on_date_and_bbox = self.fire_pixels[self.fire_pixels.acq_date.eq(fire_date)]
        unique_time = self.fire_data_filter_on_date_and_bbox.acq_time.unique()
        return unique_time

    def hhmm_to_minutes(self,hhmm):
        """Convert HHMM format to minutes since midnight."""
        hours = hhmm // 100
        minutes = hhmm % 100
        return hours * 60 + minutes
    
    def get_close_dates(self,unique_time):
        if(len(unique_time) <2):
            return []
        unique_time = np.sort(unique_time)
        unique_time_minutes = np.array([self.hhmm_to_minutes(t) for t in unique_time])
        time_intervel = np.diff(unique_time_minutes)
        min_time_intervel = time_intervel.min()
        short_unique=[]
        for i,j in enumerate(time_intervel):
            if time_intervel[i] < 10:
                short_unique.append(unique_time[i])
                short_unique.append(unique_time[i+1])
        # print(short_unique)
        short_unique = np.unique(short_unique)
        return short_unique
    
    def collapse_close_dates(self,unique_time):
            if(len(unique_time) <2):
                return unique_time
            unique_time = np.sort(unique_time)
            unique_time_minutes = np.array([self.hhmm_to_minutes(t) for t in unique_time])
            time_intervel = np.diff(unique_time_minutes)
            time_intervel = np.append(time_intervel,2400)
            short_unique=[]
            for i,j in enumerate(time_intervel):
                # if time_intervel[i] < 10:
                #     short_unique.append(unique_time[i+1])
                if time_intervel[i] > 10:
                    short_unique.append(unique_time[i])
            return short_unique
    
    def make_tiff(self, fire_date,ac_time,viirs_tif_dir = None):

        # output file name
        if viirs_tif_dir is None:
            viirs_tif_dir = viirs_dir.replace('$LOC', self.location) 

        # VIIRS_file_name = self.satellite + '-' + str(fire_date) + "_" + str(ac_time) + '.tif'
        VIIRS_file_name = VIIRS_tiff_file_name.format(fire_date = str(fire_date),
                                                       ac_time = str(ac_time))
        out_file = viirs_tif_dir + VIIRS_file_name

        if ((not VIIRS_OVERWRITE) and file_exists(out_file)):
            return
        # filter firepixel for time of date
        fire_data_filter_on_time = self.fire_data_filter_on_date_and_bbox[
            (self.fire_data_filter_on_date_and_bbox.acq_time.lt(ac_time+1)) & (self.fire_data_filter_on_date_and_bbox.acq_time.gt(ac_time-10))]
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
        for k in range(fire_data_filter_on_timestamp.shape[0]):
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
