"""
Functions to orthorectify GOES-R ABI images using a DEM
"""

import os

from matplotlib import pyplot as plt
import numpy as np
from osgeo import gdal
from osgeo import osr
import xarray as xr

from goes_ortho.rad import goesBrightnessTemp, goesReflectance


# from DEM_Map import get_artho_coff, get_dem, make_ortho_map
from CommonFunctions import prepareDirectory
from DEM_Map import DEM_Map
from SiteInfo import SiteInfo
from os.path import exists as file_exists
import numpy as np

import numpy as np

def resample_raster(site, abi_image):
    # Assuming abi_image['tb'] is 2D, and abi_image['longitude'] and abi_image['latitude'] are 1D arrays
    data_array = abi_image['tb'].values  # 2D array of values (fire data)
    y = abi_image['longitude'].values    # 1D array of longitudes
    x = abi_image['latitude'].values     # 1D array of latitudes

    # Create a 2D meshgrid from 1D arrays
    lon_grid, lat_grid = np.meshgrid(y, x)

    # Flatten the 2D arrays
    fire_lats = lat_grid.ravel()    # Flattened latitudes (same size as data_array)
    fire_lons = lon_grid.ravel()    # Flattened longitudes (same size as data_array)
    fire_values = data_array.ravel()  # Flattened data array (fire data)

    # Prepare empty raster grid
    b1_pixels = np.zeros(site.image_size, dtype=float)
    nx, ny = site.image_size[1], site.image_size[0]
    
    # Define the bounding box
    xmin, ymin, xmax, ymax = (
        site.transformed_bottom_left[0], site.transformed_bottom_left[1], 
        site.transformed_top_right[0], site.transformed_top_right[1]
    )
    
    # Vectorized transformation of lat/lon to UTM coordinates
    lon_points, lat_points = site.transformer.transform(fire_lats,fire_lons)

    # Calculate pixel coordinates (vectorized)
    cord_x = np.round((lon_points - xmin) / site.res).astype(int)
    cord_y = np.round((lat_points - ymin) / site.res).astype(int)

    # Mask out points that fall outside the raster bounds
    valid_mask = (cord_x >= 0) & (cord_x < nx) & (cord_y >= 0) & (cord_y < ny)

    # Apply the mask to the pixel coordinates and fire values
    cord_x_valid = cord_x[valid_mask]
    cord_y_valid = cord_y[valid_mask]
    fire_values_valid = fire_values[valid_mask]

    # Update the raster grid using vectorized max operation
    np.maximum.at(b1_pixels, (ny - 1 - cord_y_valid, cord_x_valid), fire_values_valid)

    return b1_pixels

def check2(site, abi_image):
    # Assuming abi_image['tb'] is 2D, and abi_image['longitude'] and abi_image['latitude'] are 1D arrays
    data_array = abi_image['tb'].values  # 2D array of values (fire data)
    y = abi_image['longitude'].values    # 1D array of longitudes
    x = abi_image['latitude'].values     # 1D array of latitudes

    # Create a 2D meshgrid from 1D arrays
    lon_grid, lat_grid = np.meshgrid(y, x)

    # Flatten the 2D arrays
    fire_lats = lat_grid.ravel()    # Flattened latitudes (same size as data_array)
    fire_lons = lon_grid.ravel()    # Flattened longitudes (same size as data_array)
    fire_values = data_array.ravel()  # Flattened data array (fire data)

    # Prepare empty raster grid
    b1_pixels = np.zeros(site.image_size, dtype=float)
    nx, ny = site.image_size[1], site.image_size[0]
    
    # Define the bounding box
    xmin, ymin, xmax, ymax = (
        site.transformed_bottom_left[0], site.transformed_bottom_left[1], 
        site.transformed_top_right[0], site.transformed_top_right[1]
    )
    
    # Vectorized transformation of lat/lon to UTM coordinates
    lon_points, lat_points = site.transformer.transform(fire_lats,fire_lons)

    # min_lon,max_lon = np.min(lon_points),np.max(lon_points)
    # min_lat,max_lat = np.min(lat_points),np.max(lat_points)
    
    # (min_lon,min_lat,max_lon,max_lat) - (xmin, ymin, xmax, ymax)
    valid_mask2 = (lon_points >= xmin) & (lon_points <= xmax) & (lat_points >= ymin) & (lat_points <=ymax)
    lon_points, lat_points = lon_points[valid_mask2], lat_points[valid_mask2]
    return b1_pixels

def resample_raster_old(site,abi_image):
        data_array = abi_image['tb'].values  # Use the correct variable
        y = abi_image['longitude'].values
        x = abi_image['latitude'].values
        fire_data_filter_on_timestamp = [(lat,lon,data_array[i][j]) for i,lat in enumerate(x) for j,lon in enumerate(y)]
        # image_size = (site.image_size[0] + 10 , site.image_size[1] +10 )
        image_size =site.image_size
        b1_pixels = np.zeros(image_size, dtype=float)
        nx = image_size[1]
        ny = image_size[0]
        xmin, ymin, xmax, ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        
        # bottom_left_utm = [int(site.transformer.transform(self.bottom_left[0], self.bottom_left[1])[0]),
        #                     int(site.transformer.transform(self.bottom_left[0], self.bottom_left[1])[1])]
        dir = {}
        for k in range(len(fire_data_filter_on_timestamp)):
            record = fire_data_filter_on_timestamp[k]
            # transforming lon lat to utm
            # lat and long should be inter change , and so does cord_x and cord_y
            lon_point = site.transformer.transform(record[0], record[1])[0]
            lat_point = site.transformer.transform(record[0], record[1])[1]
            # if (lon_point <xmin or lat_point <ymin):
            #     continue
            # if (lon_point > xmax or lat_point > ymax):
            #     continue
            cord_x = round((lon_point - xmin) / site.res)
            cord_y = round((lat_point - ymin) / site.res)
            if cord_x >= nx or cord_y >= ny:
                continue
            if (cord_x <0 or cord_y <0):
                continue
            if (cord_x ==0 or cord_y ==0):
                dir[(cord_x,cord_y)] =dir.get((cord_x,cord_y),0) + 1
                # print(cord_y, cord_x)
            #     pass
                # b1_pixels[-cord_y, cord_x] = 400
            b1_pixels[ny -1 -cord_y, cord_x] = max(b1_pixels[ny -1 -cord_y, cord_x], record[2])
            # b1_pixels[ny -1 -cord_y, cord_x] = ny -1  - cord_y
        print(dir)
        return b1_pixels


def create_raster_array(site,fire_data_filter_on_timestamp):
        b1_pixels = np.zeros(site.image_size, dtype=float)
        nx = site.image_size[1]
        ny = site.image_size[0]
        xmin, ymin, xmax, ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        
        # bottom_left_utm = [int(site.transformer.transform(self.bottom_left[0], self.bottom_left[1])[0]),
        #                     int(site.transformer.transform(self.bottom_left[0], self.bottom_left[1])[1])]
        for k in range(len(fire_data_filter_on_timestamp)):
            record = fire_data_filter_on_timestamp[k]
            # transforming lon lat to utm
            lon_point = site.transformer.transform(record[0], record[1])[0]
            lat_point = site.transformer.transform(record[0], record[1])[1]
            cord_x = round((lon_point - xmin) / site.res)
            cord_y = round((lat_point - ymin) / site.res)
            if (cord_x <0 or cord_y <0):
                continue
            if cord_x >= nx or cord_y >= ny:
                continue
            b1_pixels[-cord_y, cord_x] = max(b1_pixels[-cord_y, cord_x], record[2])
        return b1_pixels

def gdal_writter(out_file, site, b_pixels):
        crs, image_size = site.EPSG, site.image_size
        xmin, ymin, xmax, ymax = [site.transformed_bottom_left[0], site.transformed_bottom_left[1], site.transformed_top_right[0], site.transformed_top_right[1]]
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, image_size[1],
            image_size[0], len(b_pixels),
            gdal.GDT_Float32)
        # transforms between pixel raster space to projection coordinate space.
        # new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        geotransform = (xmin, 375, 0, ymax, 0, -375)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(crs)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        for i in range(len(b_pixels)):
            dst_ds.GetRasterBand(i+1).WriteArray(b_pixels[i]) 
        # dst_ds.GetRasterBand(2).WriteArray(b1_pixels[1])
        dst_ds.FlushCache()  # write to disk
        dst_ds = None    
    
def plot_verification_TP(location, fire_date, ac_time, out_file_name, resampled_fire_dat):
    # Load the GOES data
    previous_goes = f'DataRepository/reference_data_areaDef_correction/{location}/GOES/ABI-L1b-RadC07ABI-L1b-RadC14ABI-L1b-RadC15/tif/GOES-{fire_date}_{ac_time}.tif'
    GOES_data = xr.open_rasterio(previous_goes)
    gd = GOES_data.variable.data[0]
    # np.array(resampled_fire_dat)
    # if(np.count_nonzero(resampled_fire_dat==0) > 0):
    #     print(location,np.count_nonzero(resampled_fire_dat==0),np.count_nonzero(gd==0))
    # return
    shape = resampled_fire_dat.shape
    X, Y = np.mgrid[0:1:complex(str(shape[0]) + "j"), 0:1:complex(str(shape[1]) + "j")]
    # Load the VIIRS data
    v_file = f'DataRepository/reference_data_areaDef_correction/{location}/VIIRS/viirs-snpp-{fire_date}_{ac_time}.tif'
    VIIRS_data = xr.open_rasterio(v_file)
    vd = VIIRS_data.variable.data[0]
    
    # Plot setup
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    vmin = None
    vmax = None
    # Plot Satpy GOES using pcolormesh
    axs[0].pcolormesh(Y, -X,gd, cmap='jet', vmin=vmin,vmax=vmax)
    axs[0].set_title('Satpy GOES')

    # Plot Ortho_rec GOES using pcolormesh
    z = axs[1].pcolormesh(Y, -X,resampled_fire_dat, cmap='jet', vmin=vmin,vmax=vmax)
    axs[1].set_title('Ortho_rec GOES')
    # cb = fig.colorbar(z, pad=0.01,ax=axs[1])
    
    # axs[2].pcolormesh(Y, -X,resampled_fire_dat-gd, cmap='jet', vmin=vmin,vmax=vmax)

    # Overlay VIIRS with Satpy GOES
    axs[2].pcolormesh(Y, -X,vd, cmap='Greys', vmin=vmin)  # VIIRS as background
    axs[2].pcolormesh(Y, -X,gd, alpha=0.6, cmap='jet', vmin=vmin,vmax=vmax)  # GOES as overlay
    axs[2].set_title('VIIRS Overlay with Satpy GOES')

    # Overlay VIIRS with Ortho_rec GOES
    axs[3].pcolormesh(Y, -X,vd, cmap='Greys', vmin=vmin)  # VIIRS as background
    axs[3].pcolormesh(Y, -X,resampled_fire_dat, alpha=0.6, cmap='jet', vmin=vmin,vmax=vmax)  # GOES as overlay
    axs[3].set_title('Overlay with Ortho_rec GOES')

    # indices = np.where(resampled_fire_dat == 0)
    # axs[3].scatter(indices[0],-indices[1],  color='red', label="Zero Values") 
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(f"{out_file_name}.png")
    plt.close()
    # print(f"Plot saved as {out_file_name}.png")
    
# change to pcolormesh
def plot_verification(location, fire_date, ac_time, out_file_name, resampled_fire_dat):
    previous_goes = f'DataRepository/reference_data_areaDef_correction/{location}/GOES/ABI-L1b-RadC07ABI-L1b-RadC14ABI-L1b-RadC15/tif/GOES-{fire_date}_{ac_time}.tif'
    GOES_data = xr.open_rasterio(previous_goes)
    gd = GOES_data.variable.data[0]
        
        
    v_file = f'DataRepository/reference_data_areaDef_correction/{location}/VIIRS/viirs-snpp-{fire_date}_{ac_time}.tif'
    VIIRS_data = xr.open_rasterio(v_file)

    vd = VIIRS_data.variable.data[0]
    vmin = 0
        # Create the figure and a 1x3 grid for subplots (2 for side-by-side and 1 for overlap)
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        # Side-by-side comparison
        # Plot the first image (gd) on the first subplot
    axs[0].imshow(gd, interpolation='none', cmap='jet', vmin = vmin)
    axs[0].set_title('Satpy GOES')

        # Plot the second image (resampled_fire_dat) on the second subplot
    axs[1].imshow(np.array(resampled_fire_dat), interpolation='none', cmap='jet', vmin = vmin)
    axs[1].set_title('Ortho_rec GOES')

        # Overlay both images on the third subplot
    axs[2].imshow(vd, interpolation='none', cmap='Greys', vmin = vmin)  # Background image
    axs[2].imshow(gd, interpolation='none', alpha=0.6, cmap='jet', vmin = vmin)  # Overlay with transparency
    axs[2].set_title('VIIRS Overlay with Satpy GOES')
        
    axs[3].imshow(vd, interpolation='none', cmap='Greys', vmin = vmin)  # Background image
    axs[3].imshow(np.array(resampled_fire_dat), interpolation='none', alpha=0.6, cmap='jet', vmin = vmin)  # Overlay with transparency
    axs[3].set_title('Overlay with Ortho_rec GOES')

        # Add colorbars if needed for better visual understanding
        # plt.colorbar(axs[0].images[0], ax=axs[0])
        # plt.colorbar(axs[1].images[0], ax=axs[1])

        # Show the plot
    plt.tight_layout()
        # plt.show()
        # plt.imshow(gd, interpolation='none')
        # plt.imshow(np.array(rasampled_fire_dat), interpolation='none')
    plt.savefig(f"{out_file_name}.png")
    print(f"{out_file_name}.png")

# def GOES_coff_check():
#     directory = {}
#     dir = 'GOES_netcdfs/'
#     g_nc_list = os.listdir(dir)
#     for g_nc in g_nc_list:
#         goes_version = g_nc.split('_')[2] 
#         try:
#             req,rpol,H,lon_0,e =get_artho_coff(dir + g_nc)
#             pair = (goes_version, req,rpol,H,lon_0)
#             directory[pair] = directory.get(pair, -1) + 1
#             # print(g_nc)
#         except:
#             continue
#         # if(goes_version == 'G18'):
#         #     # print(goes_version)
#         #     print(goes_version, req,rpol,H,lon_0)
#         # dir[()]
#     print(directory)
#     # {('G18', 6378137.0, 6356752.31414, 42164160.0, -137.0): 9055,
#         #  ('G17', 6378137.0, 6356752.31414, 42164160.0, -137.0): 14506,
#         #  ('G16', 6378137.0, 6356752.31414, 42164160.0, -75.0): 4568}


def orthorectify_abi(goes_filepath, pixel_map, data_vars, out_filename=None):
    """
    Using the pixel mapping for a specific ABI viewing geometry over a particular location,
    orthorectify the ABI radiance values and return an xarray dataarray with those values.

    Parameters
    ------------
    goes_filepath : str
        filepath to GOES ABI NetCDF file
    pixel_map : xarray.Dataset
        dataset of the map relating ABI Fixed Grid coordinates to latitude and longitude
    data_vars : list
        list of variable names from the GOES ABI NetCDF file we wish to extract
    out_filename : str
        optional filepath and filename to save the orthorectified image to, defaults to None

    Returns
    ------------
    pixel_map : xarray.Dataset
        dataset of the orthorectified GOES ABI image

    Examples
    ------------

    """
    # print("\nRUNNING: orthorectify_abi_rad()")

    # First check, Does the projection info in the image match our mapping?
    # print("\nDoes the projection info in the image match our mapping?")
    # Open the GOES ABI image
    # print("\nOpening GOES ABI image...\t\t\tABI image value\tPixel map value")
    abi_image = xr.open_dataset(goes_filepath, decode_times=False)
    

    # Map (orthorectify) and clip the image to the pixel map for each data variable we want
    for var in data_vars:
        # print(
        #     "\nMap (orthorectify) and clip the image to the pixel map for {}".format(
        #         var
        #     )
        # )
        abi_var_values = abi_image.sel(
            x=pixel_map.dem_px_angle_x, y=pixel_map.dem_px_angle_y, method="nearest"
        )[var].values
        # print("...done")

        # Create a new xarray dataset with the orthorectified ABI radiance values,
        # Lat, Lon, Elevation, and metadata from the pixel map.
        pixel_map[var] = (["latitude", "longitude"], abi_var_values)
        # If we are looking at an ABI-L1b-Rad product, create either a reflectance (bands 1-6) or brightness temperautre (bands 7-16) dataset
        if var == "Rad":
            # if we are looking at bands 1-6, compute reflectance
            if abi_image.band_id.values[0] <= 6:
                pixel_map["ref"] = goesReflectance(
                    pixel_map[var], abi_image.kappa0.values
                )
            # else, compute brightness temperature for bands 7-16
            else:
                pixel_map["tb"] = goesBrightnessTemp(
                    pixel_map[var],
                    abi_image.planck_fk1.values,
                    abi_image.planck_fk2.values,
                    abi_image.planck_bc1.values,
                    abi_image.planck_bc2.values,
                )

    # Map (orthorectify) the original ABI Fixed Grid coordinate values to the new pixels for reference
    # print(
    #     "\nMap (orthorectify) and clip the image to the pixel map for ABI Fixed Grid coordinates"
    # )
    abi_fixed_grid_x_values = abi_image.sel(
        x=pixel_map.dem_px_angle_x.values.ravel(), method="nearest"
    ).x.values
    abi_fixed_grid_y_values = abi_image.sel(
        y=pixel_map.dem_px_angle_y.values.ravel(), method="nearest"
    ).y.values
    abi_fixed_grid_x_values_reshaped = np.reshape(
        abi_fixed_grid_x_values, pixel_map.dem_px_angle_x.shape
    )
    abi_fixed_grid_y_values_reshaped = np.reshape(
        abi_fixed_grid_y_values, pixel_map.dem_px_angle_y.shape
    )
    pixel_map["abi_fixed_grid_x"] = (
        ("latitude", "longitude"),
        abi_fixed_grid_x_values_reshaped,
    )
    pixel_map["abi_fixed_grid_y"] = (
        ("latitude", "longitude"),
        abi_fixed_grid_y_values_reshaped,
    )
    # print("...done")

    # drop DEM from dataset
    # pixel_map = pixel_map.drop(['elevation'])

    # print(
    #     "\nCreate zone labels for each unique pair of ABI Fixed Grid coordinates (for each orthorectified pixel footprint)"
    # )
    # Found this clever solution here: https://stackoverflow.com/a/32326297/11699349
    # Create unique values for every "zone" (the GOES ABI pixel footprints) with the same ABI Fixed Grid X and Y values
    unique_values = (
        pixel_map.abi_fixed_grid_x.values
        * (pixel_map.abi_fixed_grid_y.values.max() + 1)
        + pixel_map.abi_fixed_grid_y.values
    )
    # Find the index of all unique values we just created
    _, idx = np.unique(unique_values, return_inverse=True)
    # Use these indices, reshaped to the original shape, as our zone labels
    zone_labels = idx.reshape(pixel_map.abi_fixed_grid_y.values.shape)
    # Add the zone_labels to the dataset
    pixel_map["zone_labels"] = (("latitude", "longitude"), zone_labels)
    # print("...done")

    # Output this result to a new NetCDF file
    # print("\nOutput this result to a new NetCDF file")
    # if out_filename is None:
    #     out_filename = abi_image.dataset_name + "_ortho.nc"
    # print("Saving file as: {}".format(out_filename))

    # pixel_map.to_netcdf(out_filename)
    # print("...done")

    return pixel_map

def buffer_boundingbox(bounds,ortho_buffer = 0.2):
    minx,miny,maxx,maxy = bounds
    return (minx - ortho_buffer, miny - ortho_buffer, maxx + ortho_buffer, maxy + ortho_buffer)

def Oretho_rectifier_operation(location,dir='DataRepository/Ortho_results_test'):

    # GOES_netcdfs/OR_ABI-L1b-RadC-M6C07_G17_s20212162131177_e20212162133561_c20212162133594.nc 2021-08-04 2129
    site = SiteInfo(location)
    site.get_image_dimention()
    longitude = site.longitude
    bounds = (site.bottom_left[1],site.bottom_left[0],site.top_right[1],site.top_right[0])
    ortho_bounds = buffer_boundingbox(bounds)
    # ortho_bounds = bounds
    
    dem_path = f"DataRepository/DEM_site_api/tuolumne_dem_{location}.tif"
    directory = f'DataRepository/Ortho_results_test/{location}/'
    prepareDirectory(directory)
    
    ortho_map = get_orth_map_for_site(dem_path, longitude, ortho_bounds)
    
    # paths = download_goes(self, fire_date, ac_time)
    # paths = ['orthorectifier_files/OR_ABI-L1b-RadC-M6C07_G17_s20212001006177_e20212001008561_c20212001008592.nc']
    
    # paths = []
    # paths = ['DataRepository/orthorectifier_files/OR_ABI-L1b-RadC-M6C07_G17_s20212162131177_e20212162133561_c20212162133594.nc']
    # fire_date, ac_time = '2021-08-04','2129'
    paths = ['GOES_netcdfs/OR_ABI-L1b-RadC-M6C07_G17_s20192972026196_e20192972028581_c20192972029052.nc']
    fire_date, ac_time = '2019-10-24','2027'
    #  2019-10-24 2027
    
    GOES_ortho_write(directory, site, ortho_map, paths, fire_date, ac_time)
        
    

def get_orth_map_for_site(dem_path, longitude, ortho_bounds):
    GOES_version = 'G16' if(longitude > -109) else 'G17'
    DEM_map = DEM_Map(dem_path)
    DEM_map.get_dem(
            demtype="SRTMGL3",
            bounds=ortho_bounds,
            proj="+proj=lonlat +datum=GRS80"
        )
    DEM_map.get_artho_coff_by_GOES_version(GOES_version)
        # create the mapping between scan angle coordinates and lat/lon given the GOES satellite position and our DEM
    ortho_map = DEM_map.make_ortho_map()
    return ortho_map


def GOES_ortho_write(directory, site, ortho_map, paths, fire_date, ac_time):
    
    resampled_fire_dat = [None] * len(paths)
    new_filename = [None] * len(paths)
    bounds = (site.bottom_left[1],site.bottom_left[0],site.top_right[1],site.top_right[0])
    out_file_name = f"{directory}/Orthorectified_GOES-{fire_date}_{ac_time}"
    # if (file_exists(f'{out_file_name}.tif')):
    #         return   
    for i,image_path in enumerate(paths):
            # create a new filename
            # new_filename[i] = f"{out_file_name}_{i}.nc"
            # specify which data variables from the original ABI product we want in our new orthorectified file
        data_vars = ["Rad"]  # I'm only selecting the Radiance product.

            # Apply the "ortho map" and save a new NetCDF file with data variables from the original file
        abi_image = orthorectify_abi(image_path, ortho_map, data_vars)
            # abi_image = xr.open_dataset(new_filename[i], decode_times=False)
        resampled_fire_dat[i] = resample_raster(site,abi_image)
        # check0 = resample_raster_old(site,abi_image)
        # check0 =check2(site,abi_image)
        # resampled_fire_dat[i] = check0
        # if(not np.array_equal(resampled_fire_dat[i] , check0)):
        #     print('issue')
        
        # data_array = abi_image['tb'].values  # Use the correct variable
        # y = abi_image['longitude'].values
        # x = abi_image['latitude'].values
        # fire_data = [(lat,lon,data_array[i][j]) for i,lat in enumerate(x) for j,lon in enumerate(y)]
        # # fire_data2 = np.column_stack((x.ravel(), y.ravel(), data_array.ravel()))
        # resampled_fire_dat[i] = create_raster_array(site,fire_data)
            # print(resampled_fire_dat[i].shape)
            
    gdal_writter(f'{out_file_name}.tif', site, resampled_fire_dat)
        
    # Ortho_gd = resampled_fire_dat[0]
    
    # Ortho_GOES_data = xr.open_rasterio(f'{out_file_name}.tif')
    # Ortho_gd = Ortho_GOES_data.variable.data[0]
    # plot_verification_TP('Kincade', fire_date, ac_time, out_file_name, Ortho_gd)
    # plot_verification(location, fire_date, ac_time, out_file_name, Ortho_gd)
 
def visual_evaluation(location,dir = 'Ortho_results_test'):
    
    
    out_path = f'{dir}_visual/{location}'
    in_path = f'{dir}/{location}'
    prepareDirectory(out_path)

    g_nc_list = os.listdir(in_path)
    for g_nc in g_nc_list:

        Ortho_GOES_data = xr.open_rasterio(f'{in_path}/{g_nc}')
        Ortho_gd = Ortho_GOES_data.variable.data[0]
        fire_date, ac_time = g_nc.replace('.tif','').replace('Orthorectified_GOES-','').split('_')
        out_file_name = f'{out_path}/{fire_date}_{ac_time}'
        # fire_date, ac_time
        # print(location, fire_date, ac_time, out_file_name)
        plot_verification_TP(location, fire_date, ac_time, out_file_name, Ortho_gd)
        # try:
        #     plot_verification_TP(location, fire_date, ac_time, out_file_name, Ortho_gd)
        # except:
        #     print("error",out_file_name)
        #     continue       
        
if __name__ == '__main__':
    
    
    
    import os
    import pandas as pd

    toExecuteSiteList = 'config/training_sites'
    data = pd.read_csv(toExecuteSiteList)
    print(toExecuteSiteList)
    locations = data["Sites"][:]
    
    
    # locations = ['Kincade']
    for location in locations:
        # Oretho_rectifier_operation(location)
        visual_evaluation(location,dir = 'DataRepository/Ortho_results')
        
        # visual_evaluation(location,dir = 'DataRepository/Ortho_results_test')
        
    
    
    
   