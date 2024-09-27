import numpy as np
import rasterio
from shapely.geometry import box
from pyproj import Transformer
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.warp import reproject

import pandas as pd
from GlobalValues import toExecuteSiteList
from SiteInfo import SiteInfo
from osgeo import gdal
from osgeo import osr
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


# Define the bounding box in WGS84 (Lon/Lat)
def get_boundingBox(longitude, latitude, rectangular_size, tif_file):
    min_lon, min_lat = longitude - rectangular_size, latitude - rectangular_size  # Bottom-left corner (southwest)
    max_lon, max_lat = longitude + rectangular_size, latitude + rectangular_size  # Top-right corner (northeast)
    
    # bottom_left = [latitude - rectangular_size, longitude - rectangular_size]
    # top_right = [latitude + rectangular_size, longitude + rectangular_size]

    california_bbox = box(min_lon, min_lat, max_lon, max_lat)
    # Step 2: Open the Raster and Get Its CRS and Bounds

    with rasterio.open(tif_file) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds

        # Print raster information
        print(f"Raster CRS: {raster_crs}")
        print(f"Raster Bounds (EPSG:5070): {raster_bounds}")


    # Step 3: Transform Bounding Box to Raster CRS (if different)
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    california_bbox_transformed = box(min_x, min_y, max_x, max_y)

    # Print the transformed bounding box    
    print(f"California Bounding Box (EPSG:4326): {california_bbox.bounds}")
    print(f"California Bounding Box (EPSG:5070): {california_bbox_transformed.bounds}")


    # Step 4: Check for Overlap with Raster
    # Create a box from the raster bounds
    raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
    if california_bbox_transformed.intersects(raster_box):
        print("Bounding box overlaps with raster.")
    else:
        print("Bounding box does NOT overlap with raster. Check CRS and bounds.")
        
    return california_bbox_transformed


def get_crop(california_bbox_transformed, tif_file):
    # Convert bounding box to GeoJSON-like format
    bbox_geojson = [california_bbox_transformed.__geo_interface__]

    with rasterio.open(tif_file) as src:
        # Crop the image
        cropped_image, cropped_transform = mask(src, bbox_geojson, crop=True)
        
        # Update metadata
        cropped_meta = src.meta.copy()
        cropped_meta.update({
            "height": cropped_image.shape[1],
            "width": cropped_image.shape[2],
            "transform": cropped_transform
        })
        
        print("Cropped Transform:", cropped_transform)
        print("Source CRS:", src.crs)
        source_crs = src.crs
        print("Cropped Image Shape:", cropped_image.shape)

        # Save the cropped image
        cropped_output_file = 'cropped_raster.tif'
        with rasterio.open(cropped_output_file, 'w', **cropped_meta) as dst:
            dst.write(cropped_image)

        print(f"Cropped raster saved to {cropped_output_file}")

        # Now reproject the cropped image to EPSG:32610
        target_crs = 'EPSG:32610'

        # Calculate the bounds of the cropped image
        xmin, ymax = cropped_transform * (0, 0)  # Top-left corner
        xmax, ymin = cropped_transform * (cropped_image.shape[2], cropped_image.shape[1])  # Bottom-right corner
        bounds = (xmin, ymin, xmax, ymax)  # (left, bottom, right, top)

        # Calculate the new transform and shape for reprojection
        new_transform, new_width, new_height = calculate_default_transform(
            src.crs,  # Source CRS
            target_crs,  # Target CRS
            cropped_image.shape[2],  # Width
            cropped_image.shape[1],  # Height
            *bounds  # Bounding box
        )

        # Prepare an array for the resampled image
        resampled_image = np.empty((cropped_image.shape[0], new_height, new_width), dtype=cropped_image.dtype)

        # Reproject the cropped image
        reproject(
            source=cropped_image,
            destination=resampled_image,
            src_transform=cropped_transform,
            src_crs=cropped_meta['crs'],
            dst_transform=new_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        
          # Update metadata for the resampled image
        resampled_meta = cropped_meta.copy()
        resampled_meta.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'crs': target_crs  # Update the CRS in the metadata
        })

        # Save the resampled image
        resampled_output_file = 'reprojected_raster.tif'
        with rasterio.open(resampled_output_file, 'w', **resampled_meta) as dst:
            dst.write(resampled_image)

        print(f"Reprojected raster saved to {resampled_output_file}")
# 'crs': CRS.from_epsg(5070)
        # Resample to target resolution of 375 meters
        # resample(cropped_transform, cropped_image, cropped_meta)
        resample(new_transform, resampled_image, resampled_meta)
        # resample2(source_crs, cropped_transform, cropped_image.shape[2], cropped_image.shape[1], cropped_image,cropped_meta)
        # resample2(target_crs, new_transform, new_width, new_height, resampled_image,resampled_meta)

def resample2( target_crs, new_transform, new_width, new_height, resampled_image,resampled_meta):
    target_resolution = 375  # meters
    current_resolution = abs(new_transform.a)  # Current resolution
    scale_factor = current_resolution / target_resolution  # Calculate scale factor

        # Calculate new dimensions for the resampled image
    new_resampled_height = int(new_height * scale_factor)  # New height
    new_resampled_width = int(new_width * scale_factor)  # New width

        # Prepare an array for the final resampled image
    final_resampled_image = np.empty((resampled_image.shape[0], new_resampled_height, new_resampled_width), dtype=resampled_image.dtype)

        # Calculate new transform for the final resampled image
    final_resampled_transform = new_transform * rasterio.Affine.scale(scale_factor, scale_factor)

        # Perform the final resampling
    reproject(
            source=resampled_image,
            destination=final_resampled_image,
            src_transform=new_transform,
            src_crs=target_crs,
            dst_transform=final_resampled_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

        # Update metadata for the final resampled image
    final_resampled_meta = resampled_meta.copy()
    final_resampled_meta.update({
            'height': new_resampled_height,
            'width': new_resampled_width,
            'transform': final_resampled_transform,
            'crs': target_crs  # Update the CRS in the metadata
        })

        # Save the final resampled image
    final_resampled_output_file = 'resampled_raster.tif'
    with rasterio.open(final_resampled_output_file, 'w', **final_resampled_meta) as dst:
        dst.write(final_resampled_image)

    print(f"Final resampled raster saved to {final_resampled_output_file}")


def resample(cropped_transform, cropped_image, cropped_meta):
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
    new_height = int(cropped_image.shape[1] / scale_factor)  # decrease the pixel count
    new_width = int(cropped_image.shape[2] / scale_factor)
    
    # new_height = site.image_size[0]
    # new_width = site.image_size[1]

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

    print(f"Original Image Shape: {cropped_image.shape}")
    print(f"Resampled Image Shape: {resampled_image.shape}")
    print(f"Original Transform: {cropped_transform}")
    print(f"Resampled Transform: {resampled_transform}")

    # Save the resampled image
    out_file = 'resampled_raster.tif'  # Output file path
    with rasterio.open(out_file, 'w', **resampled_meta) as dst:
        dst.write(resampled_image)
    # xmin, ymin, xmax, ymax = calculate_bounds(resampled_transform, new_height, new_width)  # Calculate bounds
    # print(cropped_meta['crs'])
    # gdal_writter(out_file, EPSG, resampled_image.shape, [resampled_image],xmin, ymax)

    print(f"Resampled raster saved to {out_file}")

def calculate_bounds(cropped_transform, new_height, new_width):
    xmin = cropped_transform.c
    ymin = cropped_transform.f - (new_height * abs(cropped_transform.e))
    xmax = cropped_transform.c + (new_width * abs(cropped_transform.a))
    ymax = cropped_transform.f
    
    return xmin, ymin, xmax, ymax

def gdal_writter(out_file, crs, image_size, b_pixels,xmin, ymax):
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            out_file, image_size[2],
            image_size[1], len(b_pixels),
            gdal.GDT_Float32)
        # transforms between pixel raster space to projection coordinate space.
        # new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        geotransform = (xmin, 375, 0, ymax, 0, -375)
        dst_ds.SetGeoTransform(geotransform)  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(crs)  # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        for i in range(len(b_pixels)):
            dst_ds.GetRasterBand(i+1).WriteArray(b_pixels[i][0]) 
        # dst_ds.GetRasterBand(2).WriteArray(b1_pixels[1])
        dst_ds.FlushCache()  # write to disk
        dst_ds = None
        
        
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
        curr_img.text(0.5, -0.1, lables[k], transform=curr_img.transAxes, ha='center', fontsize=12)

        cb = fig.colorbar(p, pad=0.01)
        cb.ax.tick_params(labelsize=11)
        cb.set_label("hello", fontsize=12)
    plt.rcParams['savefig.dpi'] = 600
    if (save_path):
        fig.savefig(save_path)
        plt.close()
    plt.show()
    
if __name__ == '__main__':
    # Step 1: Define Bounding Box for California in WGS84 (EPSG:4326)
    

    tif_file = 'Elevation/LC20_Elev_220.tif'
    
    data = pd.read_csv(toExecuteSiteList)
    locations = data["Sites"][:1]
    
    for location in locations:
        site = SiteInfo(location)
        
        latitude, longitude = site.latitude, site.longitude
        rectangular_size = site.rectangular_size
        print(site.EPSG)
        
        california_bbox_transformed = get_boundingBox(longitude, latitude, rectangular_size, tif_file)

        get_crop(california_bbox_transformed,tif_file)
        
        out_file = ['resampled_raster.tif','reprojected_raster.tif','cropped_raster.tif']
        # out_file = ['resampled_raster.tif','cropped_raster.tif']
        import xarray as xr
        target = 'outp.png'
        vd = []
        label =[]
        for fileName in out_file:
            ELEVATION_data = xr.open_rasterio(fileName)
            elevation = ELEVATION_data.variable.data[0]
            vd.append(elevation)
            label.append("hello")
        if(vd):
            vmin ,vmax = None, None
            # vmin,vmax=0,3500
            Plot_list( 'title', vd, label, vd[0].shape, vmin, vmax, save_path=target)
            print(target)
    
    