import matplotlib as mpl
mpl.use('TkAgg')
# from mpl_toolkits.basemap import Basemap, cm
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import cartopy.io.shapereader as shpreader
import datetime as dt
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
    proj = ccrs.PlateCarree()

    # The projection keyword determines how the plot will look
    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=proj)

    cmap = 'YlOrRd'
    bbox = [-122, -119, 39.5, 41.5]
    ax.add_image(StreetmapESRI(), 10)
    ax.set_extent(bbox)
    reader = shpreader.Reader("DL_FIRE_SV-C2_272924/fire_archive_SV-C2_272924.shp")
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

    prd = dt.datetime.strftime(pdate, '%Y%m%d_%H%M')

    plt.savefig('plots' + '/FRP_' + prd + '.png', bbox_inches='tight',
                dpi=120)
    print('Done.')
    plt.close('all')


def show_plot4():
    # Read shape file
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent([-125, -118, 38, 42.5])

    # reader = shpreader.Reader("DL_FIRE_SV-C2_270448/fire_archive_SV-C2_270448.shp")
    reader = shpreader.Reader("DL_FIRE_SV-C2_272924/fire_archive_SV-C2_272924.shp")
    # points = list(reader.geometries())
    # 2021-08-30 , 2021-09-15
    confidence = list(map(lambda x: x.attributes.get('ACQ_DATE'), reader.records()))
    print(confidence[0])

    readeron = list(filter(lambda x: x.attributes.get('CONFIDENCE') == 'h'
                                     and x.attributes.get('ACQ_DATE') == '2021-08-05'
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
