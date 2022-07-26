import datetime as dt

import pandas as pd

from Exploration.awsDownload import download_goes
from Exploration.common_functions import get_boundingBox
from Exploration.keyvalues import GOES_TEMP
from Exploration.longLatProj import lat_lon_reproj
from Exploration.plotGoesViirs import show_plot

FDC="ABI-L2-FDCC"
RAD="ABI-L1b-RadC"
CMI="ABI-L2-CMIPC"

def draw_plot(date_list, bbox):
    # date_list = ['2021-07-23', '2021-07-24', '2021-08-02', '2021-08-05', '2021-08-05', '2021-08-05', '2021-08-06']
    # hms = ['2100', '2042', '2118', '0854', '1036', '2018', '2142']
    date_list = ['2021-08-05']
    hms = ['2018']

    for i, fdt in enumerate(date_list):
        # map_date = dt.datetime.strftime(fdt, '%Y-%m-%d')
        # print(map_date,unique_time(map_date, bbox))
        # map_date = fdt
        # map_date = dt.datetime.strftime(fdt, '%Y-%m-%d')
        # hm = unique_time(map_date, bbox)

        map_date, hm = fdt, [hms[i]]
        for hmr in hm:
            # for bd in range(1, 17):
            try:
                download_goes(map_date, hmr[:2], hmr[2:],product=FDC,band=None)
                lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_units, var_name = lat_lon_reproj(
                    GOES_TEMP, "Temp")
                show_plot(lon, lat, data, data_units, data_time_grab, data_long_name, band_id,
                          band_wavelength, band_units, var_name, bbox, map_date, hmr, save=False)
            except KeyError as e:
                print("KeyError", e, "for", map_date, hmr)


sDATE = dt.datetime.strptime("2021-07-15", '%Y-%m-%d')
eDATE = dt.datetime.strptime("2021-08-15", '%Y-%m-%d')
bboxm = get_boundingBox("dixie2")
date_list = pd.date_range(sDATE, eDATE, freq='1D')[:-1]
draw_plot(date_list, bboxm)
