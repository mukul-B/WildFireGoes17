# Script to calculate the NOAA satellites angle of view
# Developed by : Majid Bavandpour
# Email address: majid.bavandpour@gmail.com


# Please update satellites element set using this url: https://celestrak.org/NORAD/elements/index.php

import csv
from skyfield.api import EarthSatellite, load, wgs84
from datetime import datetime, timezone,timedelta
import sys
class AngleOfview:
    def __init__(self, satName):
        """
    The function calculates the angle of view for three Satellite including "NOAA 20 (JPSS-1)", "NOAA 21 (JPSS-2)", "SUOMI NPP"

    input parameters:
    satName(str)              :: satellite name which is one of these: "NOAA 20 (JPSS-1)", "NOAA 21 (JPSS-2)", "SUOMI NPP"
    dateTime(datetime object) :: datetime of the data that you want the satellite angle of view
    dataPointPosition(list)   :: a list object containing lat and lon of the data that you want the satellite angle of view [lat, lon]

    function output           :: satellite angle of view in degrees
    """

    # CSVs = {"NOAA 20 (JPSS-1)":"VIIRS_angle_view_files/sat000043013.csv",
    #         "NOAA 21 (JPSS-2)":"VIIRS_angle_view_files/sat000054234.csv",
    #         "SUOMI NPP":"VIIRS_angle_view_files/sat000037849.csv" }
    
        CSVs = {"NOAA 20 (JPSS-1)":"DataRepository/VIIRS_angle_view_files/sat000043013.csv",
            "NOAA 21 (JPSS-2)":"DataRepository/VIIRS_angle_view_files/sat000054234.csv",
            "SUOMI NPP":"DataRepository/VIIRS_angle_view_files/sat000037849.csv",
                "GOES-16":"DataRepository/VIIRS_angle_view_files/sat000041866.csv",
            "GOES-17":"DataRepository/VIIRS_angle_view_files/sat000043226.csv",
            "GOES-18":"DataRepository/VIIRS_angle_view_files/sat000051850.csv" }


        # Open the satellites element set
        with load.open(CSVs[satName], mode='r') as f:
            data = list(csv.DictReader(f))
        self.data = data
    
    def set_date(self,dateTime):
        # Generating time object
        ts = load.timescale()
        self.time = ts.from_datetime(dateTime)

        # Selecting the data using date time
        dateTimes = [EarthSatellite.from_omm(ts, fields) for fields in self.data]
        by_date = {datetime.strptime(dateTime.epoch.utc_strftime(), '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc): dateTime for dateTime in dateTimes}

        self.satellite = by_date[min(by_date.keys(), key=lambda dt: abs(dt - dateTime))]
        
    def calculateAngleOfView(self, dataPointPosition):

        # Calculating the position of the satellite
        bluffton = wgs84.latlon(dataPointPosition[0], dataPointPosition[1])
        difference = self.satellite - bluffton
        topocentric = difference.at(self.time)   

        # satLat, satLon = wgs84.latlon_of(geocentric)
        # satElev = wgs84.height_of(geocentric)
        if topocentric.altaz()[0].degrees<0:
            return -1
            # print(topocentric.altaz()[0].degrees)
            # sys.exit("The satellite is under the horizon!!!!")

        return (90-topocentric.altaz()[0].degrees)

# # 	45.674610	-108.821419	333.06	0.61	0.71	2021-01-01	0800	N	VIIRS	n	1	265.06	10.18	N	0	2021-01-01 08:00:00+00:00	86.470289
# date_str = "2022-04-06_0906"
# # lat, lon = 45.674,-108.821
# lat , lon = 36.0951470691022, -105.36972019126934
# date_time_utc = datetime.strptime(date_str, "%Y-%m-%d_%H%M").replace(tzinfo=timezone.utc)
# # GOES-16
# # 35.71782 -105.399
# angleOfView = AngleOfview('GOES-16')
# for i in range(35):
#     date_time_utc = date_time_utc + timedelta(hours = 8)
#     angleOfView.set_date(date_time_utc)
#     snpp_angle = angleOfView.calculateAngleOfView([lat, lon])
#     print(str(date_time_utc),snpp_angle)
# noaa_angle = calculateAngleOfView('SUOMI NPP',date_time_utc,[lat, lon])
# print(noaa_angle)

# def getAngleOfView(satName, dataPointPosition):
#     date_str = "2021-01-01_0906"
#     date_time_utc = datetime.strptime(date_str, "%Y-%m-%d_%H%M").replace(tzinfo=timezone.utc)
#     return calculateAngleOfView(satName, date_time_utc, dataPointPosition)
    