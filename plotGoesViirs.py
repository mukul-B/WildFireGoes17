import matplotlib as mpl

from keyvalues import viirs_file

# mpl.use('Agg')
# mpl.use('TkAgg')

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


def show_plot(lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_units,
              var_name, bbox,
              fildate, filtime=None,save = True):

    if save:
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')

    proj = ccrs.PlateCarree()
    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=proj)
    bbox2 = [bbox[0], bbox[2], bbox[1], bbox[3]]
    ax.add_image(StreetmapESRI(), 10)
    ax.set_extent(bbox)

    reader = shpreader.Reader(viirs_file, bbox2)
    readeron = list(filter(lambda x:
                           # x.attributes.get('CONFIDENCE') != 'n'
                           # and
                           x.attributes.get('ACQ_DATE') == fildate
                           and ((x.attributes.get('ACQ_TIME') == filtime) if filtime else True)
                           , reader.records()))
    points = [i.geometry for i in readeron]

    p = ax.pcolormesh(lon.data, lat.data, data,
                      transform=ccrs.PlateCarree(),
                      alpha=0.9)
    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        transform=proj,
        c="red",
        alpha=0.17
    )
    cbar = plt.colorbar(p, shrink=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False

    plt.title('{0}\n{2}{3}{4} on {1} with viirs {5}'.format(data_long_name, data_time_grab, band_id, band_wavelength,
                                                            band_units, filtime))
    # plt.tight_layout()

    pdate = dt.datetime.strptime(data_time_grab, '%Y-%m-%d %H:%M:%S.%f')
    prd = dt.datetime.strftime(pdate, '%Y%m%d_%H%M')

    if (save):
        plt.savefig('plots' + '/FRP_' + prd + '.png', bbox_inches='tight', dpi=240)
    else:
        plt.show()
    print('Done.')

    plt.close('all')
