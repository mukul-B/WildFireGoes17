import os
from matplotlib import pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
from pyproj import Transformer
import numpy as np
import pandas as pd
from GlobalValues import toExecuteSiteList
from SiteInfo import SiteInfo
import xarray as xr

import time
import multiprocessing as mp
from tqdm import tqdm
from os.path import exists as file_exists


def on_success(location):
    pbar.update(1)
    # print(f"{location}" )
    # pass
    print(f"{location} processed successfully at {time.time() - start_time:.2f} seconds")

def on_error(e):
    print(f"Error: {e}")
    
    
    
# Define the bounding box in WGS84 (Lon/Lat)
def get_boundingBox(longitude, latitude, rectangular_size, target_crs):
    min_lon, min_lat = longitude - rectangular_size, latitude - rectangular_size  # Bottom-left corner (southwest)
    max_lon, max_lat = longitude + rectangular_size, latitude + rectangular_size  # Top-right corner (northeast)
    
    # Define bounding box in WGS84 (EPSG:4326)
    california_bbox = box(min_lon, min_lat, max_lon, max_lat)

    # Step 2: Transform Bounding Box to Target CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    california_bbox_transformed = box(min_x, min_y, max_x, max_y)

    # Print the transformed bounding box    
    # print(f"Bounding Box (EPSG:4326): {california_bbox.bounds}")
    # print(f"Bounding Box ({target_crs}): {california_bbox_transformed.bounds}")
    
    return california_bbox_transformed

def reproject_raster(tif_file, target_crs,location,out_path):
    with rasterio.open(tif_file) as src:
        # Calculate the new transform and shape for reprojection
        new_transform, new_width, new_height = calculate_default_transform(
            src.crs,  # Source CRS
            target_crs,  # Target CRS
            src.width,  # Width
            src.height,  # Height
            *src.bounds  # Bounding box
        )

        # Prepare an array for the resampled image
        resampled_image = np.empty((src.count, new_height, new_width), dtype=src.dtypes[0])

        # Reproject the entire image
        reproject(
            source=rasterio.band(src, 1),
            destination=resampled_image,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=new_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        
        # Update metadata for the resampled image
        resampled_meta = src.meta.copy()
        resampled_meta.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'crs': target_crs  # Update the CRS in the metadata
        })

        # # Save the reprojected raster
        resampled_output_file = f'{out_path}/Extended_reprojected_raster_{location}.tif'
        with rasterio.open(resampled_output_file, 'w', **resampled_meta) as dst:
            dst.write(resampled_image)

        # print(f"Reprojected raster saved to {resampled_output_file}")
    
    return resampled_output_file, new_transform, resampled_image.shape

def crop_reprojected_raster(california_bbox_transformed, reprojected_file,level,location,out_path,final_image_size):
    bbox_geojson = [california_bbox_transformed.__geo_interface__]
    
    with rasterio.open(reprojected_file) as src:
        # Crop the image
        cropped_image, cropped_transform = mask(src, bbox_geojson, crop=True)
        
        # Update metadata
        cropped_meta = src.meta.copy()
        cropped_meta.update({
            "height": cropped_image.shape[1],
            "width": cropped_image.shape[2],
            "transform": cropped_transform
        })
        
        # print("Cropped Transform:", cropped_transform)
        # print("Cropped Image Shape:", cropped_image.shape)

    if level == 1 :
        # Save the cropped image
        cropped_output_file = f'{out_path}/Extended_crop_{location}.tif'
        with rasterio.open(cropped_output_file, 'w', **cropped_meta) as dst:
            dst.write(cropped_image)

        # print(f"Cropped raster saved to {cropped_output_file}")
        return cropped_output_file ,cropped_transform,cropped_meta
    
    if level == 2 :
        ret =  resample(cropped_transform, cropped_image, cropped_meta,location,out_path,final_image_size)
        return ret,None,None


def Plot_list( title, to_plot, lables, shape, vmin=None, vmax=None, save_path=None):

    
    n = len(to_plot)
    fig, ax = plt.subplots(1, n, constrained_layout=True, figsize=(4*n, 4))
    fig.suptitle(title)

    for k in range(n):
        curr_img = ax[k] if n > 1 else ax
        X, Y = np.mgrid[0:1:complex(str(to_plot[k].shape[0]) + "j"), 0:1:complex(str(to_plot[k].shape[1]) + "j")]
        p = curr_img.pcolormesh(Y, -X, to_plot[k], cmap="terrain", vmin=vmin, vmax=vmax)
        curr_img.tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
        curr_img.text(0.5, -0.1, f'{lables[k]}_{to_plot[k].shape}', transform=curr_img.transAxes, ha='center', fontsize=12)

        cb = fig.colorbar(p, pad=0.01)
        cb.ax.tick_params(labelsize=11)
        cb.set_label("hello", fontsize=12)
    plt.rcParams['savefig.dpi'] = 600
    if (save_path):
        fig.savefig(save_path)
        plt.close()
    plt.show()


def resample(cropped_transform, cropped_image, cropped_meta,location,out_path,final_image_size):
    # Target resolution (in meters)
    target_resolution = 375  # meters
    # EPSG  = 5070
    # crs = f'EPSG:{EPSG}'
    crs = cropped_meta['crs']
    
  
    # Current resolution from the Affine transform (e.g., 30 meters per pixel)
    current_resolution = abs(cropped_transform.a)  # Get the pixel size (resolution)

    # Calculate the scaling factor (target resolution / current resolution)
    scale_factor = target_resolution / current_resolution

    # Calculate new dimensions for the resampled image
    # new_height = int(cropped_image.shape[1] / scale_factor)  # decrease the pixel count
    # new_width = int(cropped_image.shape[2] / scale_factor)
    
    new_height = final_image_size[0]
    new_width = final_image_size[1]

    # Prepare an empty array for the resampled image
    resampled_image = np.empty((cropped_image.shape[0], new_height, new_width), dtype=cropped_image.dtype)

    # Update the transform to match the new resolution
    resampled_transform = cropped_transform * rasterio.Affine.scale(scale_factor, scale_factor)
    # resampled_transform= site.transformer

    # Perform the resampling using rasterio.warp.reproject
    reproject(
        source=cropped_image,
        destination=resampled_image,
        src_transform=cropped_transform,
        src_crs=cropped_meta['crs'],
        dst_transform=resampled_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear
    )

    # Update the metadata with the new dimensions and transform
    resampled_meta = cropped_meta.copy()
    resampled_meta.update({
        'height': new_height,
        'width': new_width,
        'transform': resampled_transform,
        'crs': crs
    })

    # print(f"Original Image Shape: {cropped_image.shape}")
    # print(f"Resampled Image Shape: {resampled_image.shape}")
    # print(f"Original Transform: {cropped_transform}")
    # print(f"Resampled Transform: {resampled_transform}")

    # Save the resampled image
    out_file = f'{out_path}/resampled_raster_{location}.tif'  # Output file path
    with rasterio.open(out_file, 'w', **resampled_meta) as dst:
        dst.write(resampled_image)
    # xmin, ymin, xmax, ymax = calculate_bounds(resampled_transform, new_height, new_width)  # Calculate bounds
    # print(cropped_meta['crs'])
    # gdal_writter(out_file, EPSG, resampled_image.shape, [resampled_image],xmin, ymax)

    # print(f"Resampled raster saved to {out_file}")
    return out_file
    

def create_elevation_dataset(tif_file, out_path, out_path_temp, location):
    
    
    out_file = f'{out_path}/resampled_raster_{location}.tif'
    if(file_exists(out_file)):
        
        # Viirs_file = os.listdir(f'DataRepository/reference_data_areaDef_correction/{location}/VIIRS/')[0]
        # VIIRS_data = xr.open_rasterio(out_file)
        # viirs = VIIRS_data.variable.data[0]
        
        # ELEVATION_data = xr.open_rasterio(out_file)
        # elevation = ELEVATION_data.variable.data[0]
        
        # if(viirs.shape != elevation.shape):
        #     print(location,viirs.shape != elevation.shape)
        
        return location

    # return location
    site = SiteInfo(location)
    out_file = []
    
            
    latitude, longitude = site.latitude, site.longitude
    rectangular_size = site.rectangular_size
    print(location)
    target_crs = f'EPSG:{site.EPSG}'
    site.get_image_dimention()
    final_image_size = site.image_size
    # target_crs = f'EPSG:4326'
            
            # Step 1: Get bounding box in the target CRS
            
    california_bbox_transformed_1 = get_boundingBox(longitude, latitude, 1, 'EPSG:5070')
            
            
    california_bbox_transformed = get_boundingBox(longitude, latitude, rectangular_size, target_crs)

            
    c1,_,_ = crop_reprojected_raster(california_bbox_transformed_1, tif_file,1,location,out_path_temp,final_image_size)
            # print(c1,'---------------------------------------')
            # out_file.append(c1)
            # Step 2: Reproject the raster
    reprojected_file, new_transform, shape = reproject_raster(c1, target_crs,location,out_path_temp)
            # out_file.append(reprojected_file)
            # print(reprojected_file,'---------------------------------------')

            # # Step 3: Crop the reprojected raster using the transformed bounding box
    cropped_image,_,_ = crop_reprojected_raster(california_bbox_transformed, reprojected_file,2,location,out_path,final_image_size)
    print(cropped_image,'---------------------------------------')
    return location

if __name__ == '__main__':
    # Step 1: Define Bounding Box for California in WGS84 (EPSG:4326)
    

    tif_file = 'DataRepository/Elevation/LC20_Elev_220.tif'
    # out_path = 'DataRepository/'
    # out_path_temp = 'DataRepository/'
    out_path = 'DataRepository/Per_site_elevation'
    out_path_temp = 'DataRepository/Per_site_elevation_temp'
    
    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"][:]
    start_time = time.time()
    
    parallel = 1
    if(parallel):
        pool = mp.Pool(8)
    # Initialize tqdm progress bar
    with tqdm(total=len(locations)) as pbar:
        results = []
        for location in locations:
            
            if(parallel):
                result = pool.apply_async(create_elevation_dataset, args=(tif_file, out_path, out_path_temp, location), 
                            callback=on_success, error_callback=on_error)
            # print(results.get())
            else:
                result = create_elevation_dataset( tif_file, out_path, out_path_temp, location)
                on_success(location)
        # results.append(result)
        if(parallel):
            pool.close()
            pool.join()

            # create_elevation_dataset( tif_file, out_path, out_path_temp, location)

        