import os
import pandas as pd
import time
import multiprocessing as mp
from tqdm import tqdm


from GlobalValues import  toExecuteSiteList, training_dir


from datetime import datetime, timezone
from pyproj import Transformer
from SiteInfo import SiteInfo
import numpy as np
from VIIRS_angleOfView import AngleOfview

from osgeo import gdal
from osgeo import osr

def create_angleOfView_file(location):
    site= SiteInfo(location)
    site.get_image_dimention()
    GOES_version = 'GOES-16' if(site.longitude > -109) else 'GOES-17'
    date_time_utc = datetime.combine(site.start_time, datetime.min.time(), tzinfo=timezone.utc)

    angleOfView = AngleOfview(GOES_version)
    angleOfView.set_date(date_time_utc)
    angle_view_array = reverse_resample_raster(site,angleOfView)
    out_path = 'DataRepository/AngleOfViewPerSite'
    file_name = f'loc_{location}.tif'
    gdal_writter(f'{out_path}/{file_name}', site, [angle_view_array])
    return location
    

def reverse_resample_raster(site,angleOfView):
    # Assuming resampled_image is the raster grid produced in resample_raster
    nx, ny = site.image_size[1], site.image_size[0]
    inverse_transformer = Transformer.from_crs(site.EPSG,4326)
    # Define the bounding box
    xmin, ymin, xmax, ymax = (
        site.transformed_bottom_left[0], site.transformed_bottom_left[1], 
        site.transformed_top_right[0], site.transformed_top_right[1]
    )
    lat_lon_array = np.zeros((ny, nx), dtype=float)
    # Iterate through each pixel in the raster grid
    for i in range(ny):
        for j in range(nx):
            # Convert pixel indices back to UTM coordinates
            utm_x = xmin + j * site.res
            utm_y = ymin + (ny - 1 - i) * site.res

            # Transform UTM coordinates back to lat/lon
            lat, lon = inverse_transformer.transform(utm_x, utm_y)
            lat_lon_array[i, j] = angleOfView.calculateAngleOfView([lat, lon])
    return lat_lon_array

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
        
def on_success(location):
    pbar.update(1)
    # print(f"{location}" )
    # pass
    print(f"{location} processed successfully at {time.time() - start_time:.2f} seconds")

def on_error(e):
    print(f"Error: {e}")
    
if __name__ == '__main__':
    
    print(f'Creating dataset for sites in {toExecuteSiteList}')
    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"][:]
    train_test = training_dir
    start_time = time.time()
    # train_test = testing_dir
    # pipeline run for sites mentioned in toExecuteSiteList
    # locations = ['Hermits Peak']
    parallel = 1
    if(parallel):
        pool = mp.Pool(8)
    # Initialize tqdm progress bar
    with tqdm(total=len(locations)) as pbar:
        results = []
        for location in locations:
            if(parallel):
                result = pool.apply_async(create_angleOfView_file, args=(location,), 
                                      callback=on_success, error_callback=on_error)
                # print(results.get())
            else:
                result = create_angleOfView_file(location)
                on_success(location)
            # results.append(result)
        if(parallel):
            pool.close()
            pool.join()
        # Print the last location processed
        # if results:
        #     last_processed = results[-1].get()  # Get the result of the last task
        #     print(f"Last location processed: {last_processed} at {time.time() - start_time:.2f} seconds")
