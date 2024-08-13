"""
This script create pojo( object) from config file foe site information like longitude and latitude
and duration of fire

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import yaml

from GlobalValues import site_conf


class SiteInfo():
    def __init__(self,location=None):

        self.location = location
        with open(site_conf, "r", encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.start_time, self.end_time = config.get(location).get('start') , config.get(location).get('end')
        self.latitude , self.longitude = config.get(location).get('latitude') , config.get(location).get('longitude')
        self.rectangular_size = config.get('rectangular_size')
        self.EPSG_formula  = config.get(location).get('EPSG')
        self.EPSG= self.coordinate2EPSG(self.latitude, self.longitude)

    def coordinate2EPSG(self,lat,lon):
        start_lon = -126
        end_lon = -60
        start_zone = 32610
        # lon_min, lon_max = -125.0, -66.93457 for united states
        for i in range(start_lon, end_lon, 6):
            if i < lon <= i + 6:
                return start_zone
            start_zone +=1
        # if -126.0 < lon <= -120.0:
        #     return 32610
        # if -120.0 < lon <= -114.0:
        #     return 32611
        # if -114.0 < lon <= -108.0:
        #     return 32612
