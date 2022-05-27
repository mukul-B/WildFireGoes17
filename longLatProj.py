#!/usr/bin/python
import os

os.environ['PROJ_LIB'] = "D:\PycharmProjects\Goes17"
from netCDF4 import Dataset
import matplotlib as mpl

mpl.use('TkAgg')
# from mpl_toolkits.basemap import Basemap, cm
import numpy as np

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import cartopy.io.shapereader as shpreader
import datetime as dt

def lat_lon_reproj(nc_folder, var_name=None):

    g16_data_file = nc_folder
    # designate dataset
    g16nc = Dataset(g16_data_file, 'r')
    var_names = [ii for ii in g16nc.variables]
    if (var_name is None):
        var_name = var_names[0]
    # var_name="Power"
    print(var_name)
    # exit(0)
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

    # data info
    data = g16nc.variables[var_name][:]
    data_units = g16nc.variables[var_name].units
    data_time_grab = ((g16nc.time_coverage_end).replace('T', ' ')).replace('Z', '')
    data_long_name = g16nc.variables[var_name].long_name

    # close file when finished
    g16nc.close()
    g16nc = None

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

    return lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_wavelength_units, var_name


def show_plot(lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_units,
              var_name):
    # print(np.min(lon), np.min(lat), np.max(lon), np.max(lat))
    # bbox = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]  # set bounds for plotting

    # bbox = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]
    bbox = [-125, 38, -118, 42.5]
    # figure routine for visualization
    fig = plt.figure(figsize=(9, 4), dpi=200)

    n_add = 0
    # m = Basemap(llcrnrlon=bbox[0] - n_add, llcrnrlat=bbox[1] - n_add, urcrnrlon=bbox[2] + n_add,
    #             urcrnrlat=bbox[3] + n_add,
    #             resolution='i', projection='cyl')
    # m.drawcoastlines(linewidth=0.5)
    # m.drawcountries(linewidth=0.25)
    # m.drawstates(linewidth=0.125)
    # # m.drawlsmask()
    # # print(lon.data)
    # m.pcolormesh(lon.data, lat.data, data, latlon=True)
    #
    # parallels = np.linspace(np.min(lat), np.max(lat), 25)
    # m.drawparallels(parallels, linewidth=0.125, labels=[True, False, False, False])
    # meridians = np.linspace(np.min(lon), np.max(lon), 35)
    # m.drawmeridians(meridians, linewidth=0.125, labels=[False, False, False, True])
    #
    # cb = m.colorbar()
    # # print("data_units",data_units)
    # data_units = ((data_units.replace('-', '^{-')).replace('1', '1}')).replace('2', '2}')
    # print("data_units", data_units)
    # plt.rc('text', usetex=True)
    # cb.set_label(r'%s' % (var_name))
    # # cb.set_label(r'%s $ \left[ \mathrm{%s} \right] $' % (var_name, data_units))
    # plt.title('{0}\n{2}{3}{4} on {1}'.format(data_long_name, data_time_grab, band_id, band_wavelength, band_units))
    # plt.tight_layout()
    #
    # # plt.savefig('goes_16_demo.png',dpi=200,transparent=True) # uncomment to save figure
    # plt.show()


from cartopy.io.img_tiles import GoogleTiles


class StreetmapESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Street_Map/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url


def show_plot2(lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_units,
               var_name,
               fildate):



    bbox = [-125, 38, -118, 42.5]
    proj = ccrs.PlateCarree()
    # n_add = 0
    # m = Basemap(llcrnrlon=bbox[0] - n_add, llcrnrlat=bbox[1] - n_add, urcrnrlon=bbox[2] + n_add, urcrnrlat=bbox[3] + n_add,
    #             resolution='i', projection='cyl')
    # m.pcolormesh(lon.data, lat.data, data, latlon=True)
    #
    # plt.tight_layout()
    # plt.show()

    # The projection keyword determines how the plot will look
    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())

    cmap = 'YlOrRd'
    bbox = [-122, -119, 39.5, 41.5]
    ax.add_image(StreetmapESRI(), 10)
    ax.set_extent(bbox)
    reader = shpreader.Reader("DL_FIRE_SV-C2_272924/fire_archive_SV-C2_272924.shp")
    # points = list(reader.geometries())
    # confidence = list(map(lambda x: x.attributes.get('CONFIDENCE'), reader.records()))
    # ax.scatter(
    #     [point.x for n, point in enumerate(points) if confidence[n] == 'h'],
    #     [point.y for n, point in enumerate(points) if confidence[n] == 'h'],
    #     transform=proj,
    #     alpha=0.25
    # # )

    readeron = list(filter(lambda x: x.attributes.get('CONFIDENCE') == 'h'
                                     and x.attributes.get('ACQ_DATE') == fildate
                           , reader.records()))
    points = [i.geometry for i in readeron]

    ax.pcolormesh(lon.data, lat.data, data,
                  transform=ccrs.PlateCarree(),
                  alpha=0.3)
    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        transform=proj,
        alpha=0.7
    )
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False

    # ax.contourf(lon.data, lat.data, data, latlon=True)  # didn't use transform, but looks ok...

    # parallels = np.linspace(np.min(lat), np.max(lat), 25)
    # m.drawparallels(parallels, linewidth=0.125, labels=[True, False, False, False])
    # meridians = np.linspace(np.min(lon), np.max(lon), 35)
    # m.drawmeridians(meridians, linewidth=0.125, labels=[False, False, False, True])
    #
    # cb = m.colorbar()
    # print("data_units",data_units)
    # data_units = ((data_units.replace('-', '^{-')).replace('1', '1}')).replace('2', '2}')
    # print("data_units", data_units)
    # plt.rc('text', usetex=True)
    # # cb.set_label(r'%s' % (var_name))
    # # cb.set_label(r'%s $ \left[ \mathrm{%s} \right] $' % (var_name, data_units))
    plt.title('{0}\n{2}{3}{4} on {1}'.format(data_long_name, data_time_grab, band_id, band_wavelength, band_units))
    # plt.tight_layout()
    #
    # # plt.savefig('goes_16_demo.png',dpi=200,transparent=True) # uncomment to save figure
    # plt.show()
    # %Y-%m-%d %H:%M:%S

    pdate = dt.datetime.strptime(data_time_grab, '%Y-%m-%d %H:%M:%S.%f')

    prd= dt.datetime.strftime(pdate, '%Y%m%d_%H%M')

    plt.savefig('plots'+'/FRP_' +prd + '.png', bbox_inches='tight',
                dpi=120)
    print('Done.')
    plt.close('all')


def show_plot3():
    # Read shape file
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    # Show only Africa
    ax.set_extent([-125, -118, 38, 42.5])
    # ax.stock_img()
    # ax.add_feature(cf.COASTLINE, lw=2)
    # Make figure larger
    # plt.gcf().set_size_inches(20, 10)

    # Filter for a specific country
    # kenya = [country for country in reader.records() if country.attributes["NAME_LONG"] == "Kenya"][0]
    # Display Kenya's shape
    # shape_feature = ShapelyFeature(
    #     [i for i in reader.geometries()],
    #     ccrs.PlateCarree(),
    #     facecolor="lime",
    #     edgecolor='black',
    #     lw=1
    # )
    reader = shpreader.Reader("DL_FIRE_SV-C2_270448/fire_archive_SV-C2_270448.shp")
    points = list(reader.geometries())
    confidence = list(map(lambda x: x.attributes.get('CONFIDENCE'), reader.records()))
    ax.scatter(
        [point.x for n, point in enumerate(points) if confidence[n] == 'h'],
        [point.y for n, point in enumerate(points) if confidence[n] == 'h'],
        transform=proj,
        alpha=0
    )
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    ax.add_image(StreetmapESRI(), 10)
    plt.show()
    # data = geopandas.read_file("DL_FIRE_SV-C2_270448/fire_archive_SV-C2_270448.shp")
    bbox = [-125, 38, -118, 42.5]
    # plt.figure(figsize=(6, 3))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    #
    # ax.add_geometries(data.geometry, crs=ccrs.PlateCarree())  # for Lat/Lon data.
    #
    # cmap = 'YlOrRd'
    # bbox = [-125, -118, 38, 42.5]
    # # ax.pcolormesh(lon.data, lat.data, data,
    # #               transform=ccrs.PlateCarree())
    # # ax.set_extent(bbox)
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.0)
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlines = False
    # gl.ylines = False
    # # ax.add_image(StreetmapESRI(), 10)
    # # ax.contourf(lon.data, lat.data, data, latlon=True)  # didn't use transform, but looks ok...
    # plt.tight_layout()
    # plt.show()


def show_plot4():
    # Read shape file
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent([-125, -118, 38, 42.5])

    # reader = shpreader.Reader("DL_FIRE_SV-C2_270448/fire_archive_SV-C2_270448.shp")
    reader = shpreader.Reader("DL_FIRE_SV-C2_272924/fire_archive_SV-C2_272924.shp")
    # points = list(reader.geometries())
    # 2021-08-30 , 2021-09-15
    confidence = list(map(lambda x: x.attributes.get('ACQ_DATE') , reader.records()))
    print(confidence[0])

    readeron = list(filter(lambda x: x.attributes.get('CONFIDENCE') == 'h'
                                     and x.attributes.get('ACQ_DATE') == '2021-08-06'
                           , reader.records()))
    # print(confidence[0])
    # print(readeron[0])
    points = [i.geometry for i in readeron]
    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        transform=proj,
        alpha=0.2
    )
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    ax.add_image(StreetmapESRI(), 10)
    plt.show()



# show_plot4()
