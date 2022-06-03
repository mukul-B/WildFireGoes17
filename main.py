import datetime as dt

import pandas as pd

from awsDownload import download_goes
from common_functions import unique_time, get_boundingBox
from keyvalues import GOES_TEMP
from longLatProj import lat_lon_reproj
from plotGoesViirs import show_plot


sDATE = dt.datetime.strptime("2021-08-03", '%Y-%m-%d')
eDATE = dt.datetime.strptime("2021-08-04", '%Y-%m-%d')

bbox = get_boundingBox("dixie2")
# date_list = pd.date_range(sDATE, eDATE, freq='1D')[:-1]
# hm = ['2200']
date_list = ['2021-07-23', '2021-07-24', '2021-08-02', '2021-08-05', '2021-08-05', '2021-08-05', '2021-08-06']
hms = ['2100', '2042', '2118', '0854', '1036', '2018', '2142']
# date_list = [ '2021-08-05']
# hms = ['2018']
i=0

for fdt in date_list:
    # map_date = dt.datetime.strftime(fdt, '%Y-%m-%d')
    # hm = unique_time(map_date, bbox)
    # print(hm)
    map_date=fdt
    hm =[hms[i]]
    i+=1
    for hmr in hm:
        try:
            download_goes(map_date, hmr[:2], hmr[2:])
            lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_units, var_name = lat_lon_reproj(
                GOES_TEMP, "Temp")
            show_plot(lon, lat, data, data_units, data_time_grab, data_long_name, band_id,
                      band_wavelength, band_units, var_name, bbox, map_date, hmr, save=False)
        except KeyError:
            print("no content for", map_date, hmr)
