import matplotlib as mpl

from common_functions import get_boundingBox
from keyvalues import viirs_file

# mpl.use('Agg')
mpl.use('TkAgg')

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import cartopy.io.shapereader as shpreader
import datetime as dt
from cartopy.io.img_tiles import GoogleTiles
import pandas as pd


class StreetmapESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Street_Map/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url


def show_plot4():
    # Read shape file

    bbox = get_boundingBox()
    bbox2= [bbox[0], bbox[2], bbox[1], bbox[3]]
    # print(bbox, bbox2)
    # exit(0)
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    # bbox = [-122, -119, 39.5, 41.5]
    # bbox2 = [-122, 39.5, -119, 41.5]
    ax.set_extent(bbox)

    # reader = shpreader.Reader("DL_FIRE_SV-C2_270448/fire_archive_SV-C2_270448.shp")
    reader = shpreader.Reader(viirs_file, bbox2)
    # points = list(reader.geometries())
    # 2021-08-30 , 2021-09-15
    # confidence = list(map(lambda x: x.attributes.get('ACQ_DATE'), reader.records()))
    # print(confidence[0])
    # (minx, miny,
    #         maxx, maxy)
    # (left, right, bottom, top)
    # xmin, xmax, ymin, ymax = extent

    readeron = list(filter(lambda x: x.attributes.get('CONFIDENCE') != 'n'
                                     and x.attributes.get('ACQ_DATE') == '2021-08-05'
                                     and x.attributes.get('ACQ_TIME') == "2018"
                           , reader.records()))
    # print(confidence[0])
    # print(readeron)
    points = [i.geometry for i in readeron]
    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        transform=proj,
        alpha=0.8
    )
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, alpha=0.0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    ax.add_image(StreetmapESRI(), 10)
    plt.show()



show_plot4()
# ['0836' '1018' '2000' '2142']
