"""
This script create pojo( object) from config file foe site information like longitude and latitude
and duration of fire

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""

import yaml

class SiteInfo():
    def __init__(self,location="dixie"):
        self.location = location
        with open("config/configuration.yml", "r", encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.start_time, self.end_time = config.get(location).get('start') , config.get(location).get('end')
        self.latitude , self.longitude = config.get(location).get('latitude') , config.get(location).get('longitude')
        self.rectangular_size = config.get('rectangular_size')