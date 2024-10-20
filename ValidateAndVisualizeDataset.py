"""
This script will have matrix to evalues GOES and VIIRS final dataset
checking dimention of image
checking signal to noise ratio
visualising them

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image
import datetime
from os.path import exists as file_exists
from GlobalValues import GOES_MAX_VAL, VIIRS_UNITS, GOES_product_size, viirs_dir, goes_dir, compare_dir
from PlotInputandResults2 import ImagePlot2, Plot_individual_list, Plot_list
# from VIIRS_angleOfView import calculateAngleOfView
from WriteDataset4DLModel import Normalize_img
from CommonFunctions import prepareDirectory
from SiteInfo import SiteInfo



def getth(image, on=0):
    # bins= 413
    # print(max([ image[i].max() for i in range(len(image))]) != image.max())
    
    bins = int(image.max()-image.min() + 1)
    # on = int(image.min())
    # Set total number of bins in the histogram
    image_r = image.copy()
    # image_r = image_r * (bins-1)
    # Get the image histogram
    hist, bin_edges = np.histogram(image_r, bins=bins)
    if (on):
        hist, bin_edges = hist[on:], bin_edges[on:]
    # Get normalized histogram if it is required

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    threshold = threshold + on if ((threshold + on) < (bins-1)) else threshold
    image_r[image_r < threshold] = 0
    image_r[image_r >= threshold] = 1
    return round(threshold, 2), image_r, hist, bin_edges, index_of_max_val

# visualising GOES and VIIRS
def viewtiff(location,v_file, g_file, date, save=True, compare_dir=None,time_independent_data={}):
    VIIRS_data = xr.open_rasterio(v_file)
    GOES_data = xr.open_rasterio(g_file)

    vd = VIIRS_data.variable.data[0]
    FRP = VIIRS_data.variable.data[1]
    gd = [GOES_data.variable.data[i] for i in range(GOES_product_size)]
    # if in GOES and VIIRS , the values are normalized, using this flag to visualize result
    normalized = False
    vmin,vmax = (0, 250) if normalized else (200,420)
    # to_plot, lables = multi_spectral_plots(vd, gd)
    # to_plot, lables = multi_spectralAndFRP(vd, FRP, gd)
    to_plot, lables = basic_plot(vd, gd)
    imges_plots = [ ImagePlot2(to_plot[i],lables[i],VIIRS_UNITS,'jet') for i in range(len(to_plot))]
    
    if('angle_of_view' in time_independent_data.keys()):
        angle_of_view = time_independent_data['angle_of_view']
        imges_plots.append(ImagePlot2(angle_of_view,'angle_of_view','Angle(degree)','jet'))
    
    if('elevation' in time_independent_data.keys()):
        elevation = time_independent_data['elevation']
        virrs_at_elevation = elevation[vd !=0]
        if(len(virrs_at_elevation) == 0):
            print("woh")
            return
        min_elevation , max_elevation = np.min(virrs_at_elevation) , np.max(virrs_at_elevation)
        # print(min_elevation , max_elevation)
        elevation_bucket = int(400 *(max_elevation // 400))
        # elevation[vd ==0] = 0
        elevation2 = np.copy(elevation)
        elevation2[vd !=0] = 0
        
        elevation_image_plot = ImagePlot2(elevation2,f'Elevation of VIIRS \n Viirs at max:{max_elevation} min:{min_elevation}','Elevation(m)','terrain')
        elevation_image_plot.vmin , elevation_image_plot.vmax =  (0 , 3500)
        imges_plots.append(elevation_image_plot)


    site = SiteInfo(location)
    longitude = site.longitude
    latitude = site.latitude
    compare_dir = compare_dir.replace( location,'')

    # small_big = "small" if np.count_nonzero(vd) < 30 else "big"
    # small_big = 'issueGOES'
    # compare_dir = compare_dir.replace('compare','compare_'+small_big)
    # compare_dir = f'{compare_dir}/{str(site.EPSG)}'
    # compare_dir = f'{compare_dir}/c{int((latitude // 4 ) * 4)}_{int(longitude)}'
    # compare_dir = f'{compare_dir}/{int(longitude)}_{int(latitude)}_{location}'
    # compare_dir = f'{compare_dir}/{location}'
    east_West = "east" if(longitude > -109) else "west"
    goes_sat = 'GOES-16' if(east_West == "east") else 'GOES-17'
    date_time_utc = datetime.datetime.strptime(date, "%Y-%m-%d_%H%M").replace(tzinfo=datetime.timezone.utc)
    # snpp_angle = calculateAngleOfView('SUOMI NPP',date_time_utc,[latitude, longitude])
    # noaa_angle = calculateAngleOfView('NOAA 20 (JPSS-1)',date_time_utc,[latitude, longitude])
    
    # GOES_angle = calculateAngleOfView(goes_sat,date_time_utc,[latitude, longitude])
    
    # snpp_angle_bucket = int(10 * (snpp_angle//10) ) if (snpp_angle != -1) else 'N'
    # noaa_angle_bucket = int(10 * (noaa_angle//10) ) if (noaa_angle != -1) else 'N'
    
    # Viirs_angle_bucket = f'{snpp_angle_bucket}_{noaa_angle_bucket}'
    
    # GOES_angle_bucket =  str(int(5 * (GOES_angle//5) ))
    
    viirs_with_angle_label = ''
    # if(snpp_angle >0):
    #     viirs_with_angle_label = viirs_with_angle_label + f'\nsnpp_angle:{int(snpp_angle) }'
    # if(noaa_angle >0):
    #     viirs_with_angle_label = viirs_with_angle_label + f'\nnoaa_angle:{int(noaa_angle) }'
    # lables[0] = lables[0] + f'\n{goes_sat}_angle:{int(GOES_angle) }'
    # lables[1] = lables[1]+ viirs_with_angle_label
    
    # compare_dir = f'{compare_dir}/{east_West}/{elevation_bucket}/{GOES_angle_bucket}'
    # compare_dir = f'{compare_dir}2/{east_West}/{elevation_bucket}/{Viirs_angle_bucket}'
    # compare_dir = f'{compare_dir}/{int(20 *(FRP.sum() // 20))}'
    # file_name = f'{int(longitude)}_{int(latitude)}_{location}_{date}.png'
    file_name = f'{location}_{date}.png'
    save_path = f'{compare_dir}/{file_name}' if save else None
    # save_path = f'{compare_dir}/{str(site.EPSG)}/{int(longitude)}_{int(latitude)}_{location}_{date}.png' if save else None
    # plot_condition = True
    plot_condition = (np.count_nonzero(vd) > 200 )
    # plot_condition = (np.count_nonzero(gd[0]==0) > 5)
    # plot_condition = (np.count_nonzero(vd) < 30 and FRP.sum() <150 )
    if (file_exists(save_path)):
            return
    
    if(plot_condition):
        prepareDirectory(compare_dir)
        plot_title = f'{location} at {date} coordinates : {longitude},{latitude}'
        Plot_list(plot_title,imges_plots, vd.shape, save_path)
        print(save_path)

def basic_plot(vd, gd):
    to_plot = [gd[0],vd,(gd[0] - vd)]
    lables = ["GOES","VIIRS","VIIRS On GOES"]
    return to_plot,lables

def multi_spectralAndFRP(vd, FRP, gd):
    Active_fire = (gd[0]-gd[1])/(gd[0]+gd[1])
    cloud_remove_280 = Active_fire * (gd[2]> 280) * 1000
    to_plot = [gd[0],vd,(gd[0] - vd),FRP,Active_fire,cloud_remove_280]
    lables = ["GOES","VIIRS "+str(vd.sum())+' '+str(np.count_nonzero(vd))+' '+str(round(np.average(vd),2)),"VIIRS On GOES",'FRP '+str(FRP.sum())+' '+str(np.count_nonzero(FRP))+' '+str(round(np.max(FRP),2)),"Active_fire",'cloud_remove_280']
    return to_plot,lables

def multi_spectral_plots(vd, gd):
    Active_fire = (gd[0]-gd[1])/(gd[0]+gd[1])
    cloud_remove_280 = Active_fire * (gd[2]> 280) * 1000
    # ret1, th1, hist1, bins1, index_of_max_val1 = getth(cloud_remove_280, on=0)
    # print(cloud_remove_280.min(),cloud_remove_280.max(),ret1)
    # cloud_remove = Active_fire * (cloud_remove_280 > ret1) * 1000
    cloud_remove = cloud_remove_280 * (cloud_remove_280 > 0) 
    cloud_remove = (cloud_remove * GOES_MAX_VAL)  / cloud_remove.max()
    gd[0] = Normalize_img(gd[0])
    cloud_remove = Normalize_img(cloud_remove,gf_min = 0, gf_max = GOES_MAX_VAL)
    to_plot = [gd[0],Active_fire,cloud_remove_280,cloud_remove,vd]
    lables = ["GOES","Active_fire","cloud_remove_280","cloud_remove","VIIRS"]
    
    # to_plot = [gd[0],Active_fire,cloud_remove_280,vd]
    # lables = ["GOES","Active_fire","Active_fire with Cloud Mask","VIIRS"]
    return to_plot,lables


def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    print(imdff)

    imdff = imdff.flatten()
    rmse = math.sqrt(np.mean(np.array(imdff ** 2)))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def shape_check(v_file, g_file):
    VIIRS_data = xr.open_rasterio(v_file)
    GOES_data = xr.open_rasterio(g_file)

    vf = VIIRS_data.variable.data[0]
    gd = GOES_data.variable.data[0]
    vf = np.array(vf)[:, :]
    gd = np.array(gd)[:, :]
    # (343,)(27, 47)
    if(vf.shape != gd.shape):
        print(vf.shape, gd.shape)
    # print(PSNR(gd, vf))


# the dataset created is evaluated visually and statistically
def validateAndVisualizeDataset(location, product):
    # product_name = product['product_name']
    # band = product['band']
    viirs_tif_dir = viirs_dir.replace('$LOC', location)
    product_band = ''.join(map(lambda item: f"{item['product_name']}{format(item['band'],'02d')}", product))
    goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD_BAND', product_band)
    # goes_tif_dir = goes_dir.replace('$LOC', location).replace('$PROD', product['product_name']).replace('$BAND', format(product['band'],'02d'))
    comp_dir = compare_dir.replace('$LOC', location)
    viirs_list = os.listdir(viirs_tif_dir)
    
    time_independent_data = {}
    plot_time_independent_data = True
    if plot_time_independent_data:
        elevation_path =  f'DataRepository/Per_site_elevation/resampled_raster_{location}.tif'
        ELEVATION_data = xr.open_rasterio(elevation_path)
        elevation = ELEVATION_data.variable.data[0]
        time_independent_data['elevation'] = elevation
        
        angle_of_view_path = f'DataRepository/AngleOfViewPerSite/loc_{location}.tif'
        angle_of_view_data = xr.open_rasterio(angle_of_view_path)
        angle_of_view = angle_of_view_data.variable.data[0]
        time_independent_data['angle_of_view'] = angle_of_view
            
    for v_file in viirs_list:
        g_file = "GOES" + v_file[10:]
        sample_date = v_file[11:-4]
        sample_date = sample_date.split('_')
        sample_date = f'{sample_date[0]}_{sample_date[1].rjust(4,"0")}'
        # shape_check(viirs_tif_dir + v_file, goes_tif_dir + g_file)
        viewtiff(location,viirs_tif_dir + v_file, goes_tif_dir + g_file, sample_date, compare_dir=comp_dir, save=True,time_independent_data=time_independent_data)
