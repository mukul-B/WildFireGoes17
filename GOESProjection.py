#!/usr/bin/python
import os

os.environ['PROJ_LIB'] = "D:\PycharmProjects\WildFireGoes17"
from netCDF4 import Dataset
import numpy as np


def lat_lon_reproj(nc_folder, var_name=None):
    g16_data_file = nc_folder
    # designate dataset
    g16nc = Dataset(g16_data_file, 'r')

    # data info
    var_names = [ii for ii in g16nc.variables]
    print(var_names)
    if var_name is None:
        var_name = var_names[0]
    print(var_name)
    data = g16nc.variables[var_name][:]
    data_units = g16nc.variables[var_name].units
    data_time_grab = ((g16nc.time_coverage_end).replace('T', ' ')).replace('Z', '')
    data_long_name = g16nc.variables[var_name].long_name

    try:
        band_id = g16nc.variables['band_id'][:]
        band_id = ' (Band: {},'.format(band_id[0])
        band_wavelength = g16nc.variables['band_wavelength']
        band_wavelength_units = band_wavelength.units
        band_wavelength_units = '{})'.format(band_wavelength_units)
        band_wavelength = ' {0:.2f} '.format(band_wavelength[:][0])
        print('Band ID: {}'.format(band_id))
        print('Band Wavelength: {} {}'.format(band_wavelength, band_wavelength_units))
    except:
        band_id = ''
        band_wavelength = ''
        band_wavelength_units = ''

    # GOES-R projection info and retrieving relevant constants
    proj_info = g16nc.variables['goes_imager_projection']
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    # grid info
    lat_rad_1d = g16nc.variables['x'][:]
    lon_rad_1d = g16nc.variables['y'][:]

    # close file when finished
    g16nc.close()
    g16nc = None

    lat, lon = lat_lon(H, lon_origin,  r_eq, r_pol ,lat_rad_1d, lon_rad_1d)

    return lon, lat, data , data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_wavelength_units, var_name


def lat_lon(H, lon_origin,  r_eq, r_pol ,lat_rad_1d, lon_rad_1d):
    # create meshgrid filled with radian angles
    lat_rad, lon_rad = np.meshgrid(lat_rad_1d, lon_rad_1d)
    # lat/lon calc routine from satellite radian angle vectors
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(lat_rad), 2.0) + (np.power(np.cos(lat_rad), 2.0) * (
            np.power(np.cos(lon_rad), 2.0) + (((r_eq * r_eq) / (r_pol * r_pol)) * np.power(np.sin(lon_rad), 2.0))))
    b_var = -2.0 * H * np.cos(lat_rad) * np.cos(lon_rad)
    c_var = (H ** 2.0) - (r_eq ** 2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var ** 2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(lat_rad) * np.cos(lon_rad)
    s_y = - r_s * np.sin(lat_rad)
    s_z = r_s * np.cos(lat_rad) * np.sin(lon_rad)
    lat = (180.0 / np.pi) * (
        np.arctan(((r_eq * r_eq) / (r_pol * r_pol)) * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))))
    lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)
    # print test coordinates
    print('{} N, {} W'.format(lat[318, 1849], abs(lon[318, 1849])))
    return lat, lon


lat_lon_reproj("OR_ABI-L1b-RadC-M6C07_G17_s20212170856177_e20212170858561_c20212170858591.nc")