import geopandas
import numpy as np
import pandas as pd

from keyvalues import viirs_file


def unique_time(date="2021-08-05", bbox=(-122, 39.5, -119, 41.5)):
    # bbox = (-122, 39.5, -119, 41.5)
    bbox2 = [bbox[0], bbox[2], bbox[1], bbox[3]]
    gdf = geopandas.read_file(viirs_file, bbox2)
    # print(gdf.columns.tolist())
    # print(gdf["DAYNIGHT"])
    uk = gdf[gdf.ACQ_DATE == date]
    uk = uk[uk.CONFIDENCE != 'l']
    # uk = gdf[gdf.DAYNIGHT == "N"]
    n = uk.ACQ_TIME
    nu = np.unique(n).tolist()
    return nu


def get_boundingBox(loc="dixie"):
    data = pd.read_csv("fireLocations")
    data = data[data.fire == loc]
    # print(data["xmin"].values[0])
    # exit(0)
    xmin = data["xmin"].values[0]
    xmax = data["xmax"].values[0]
    ymin = data["ymin"].values[0]
    ymax = data["ymax"].values[0]
    bbox = [xmin, xmax, ymin, ymax]
    # bbox2 = [xmin, ymin, xmax, ymax]
    # for shpreader
    return bbox


# print(unique_time("2022-06-05"))
# ['1942', '1948', '1954', '2000', '2006', '2012', '2018', '2024', '2030', '2036', '2042', '2048', '2054', '2100', '2106', '2112', '2118', '2124', '2130', '2136', '2142', '2148', '2154', '2200', '2206']
# ['0836', '0842', '0848', '0854', '0900', '0906', '0912', '0918', '0924', '0930', '0936', '0942', '0948', '0954', '1000', '1006', '1012', '1018', '1024', '1030', '1036', '1042', '1048', '1054', '1100', '1106', '1112']
