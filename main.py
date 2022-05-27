from awsDownload import download_goes
from longLatProj import lat_lon_reproj
from plotGoesViirs import show_plot2

file = "goestemp/files/satfile2.nc"

map_dat="2021-08-0"
hour="04"
min="23"

for day in range(4,5):
    map_date=map_dat + str(day)
    print(map_date)
    download_goes(map_date,hour,min)
    lon, lat, data, data_units, data_time_grab, data_long_name, band_id, band_wavelength, band_units, var_name = lat_lon_reproj(
        file, "Temp")
    show_plot2(lon, lat, data, data_units, data_time_grab, data_long_name, band_id,
               band_wavelength, band_units, var_name,map_date)