"""
This script contains RadarProcessing class contain functionality

Created on Sun june 23 11:17:09 2023

@author: mukul
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class RadarProcessing:
    def __init__(self):
        self.d = 1

    def plot_radar_csv(self, file_r, ax):
        listx = []
        listy = []
        data = pd.read_csv(file_r, header=None)
        for ind in range(data.shape[1]):
            listx.append(data[ind][0])
            listy.append(data[ind][1])
        ax.plot(listx, listy)

    def read_json_perim(self, filename):
        fjson = filename

        try:
            with open(fjson) as f:
                gj_f = json.load(f)['features']
        except:
            return None
        gj = [i for i in gj_f if i['geometry'] is not None]
        if len(gj) == 0:
            return None
        if gj[0]['geometry']['type'] == 'MultiPolygon':
            mpoly = gj[0]['geometry']['coordinates']
            perim = []
            if len(mpoly) == 1:
                mpoly = mpoly[0]
            for kk, ii in enumerate(mpoly):
                perim.append(np.squeeze(np.array(ii)))
        else:
            perim = []
            for kk, ii in enumerate(gj):
                perim.append(np.squeeze(np.array(ii['geometry']['coordinates'])))
        return perim

    def plot_radar_json(self, file_r, ax):
        perim = self.read_json_perim(file_r)
        if (not perim):
            return None
        for peri_p in perim:
            if not peri_p.shape[0] <= 2:
                # radar_poly = Polygon([(i[0], i[1]) for i in peri_p])
                # ax.fill(*radar_poly.exterior.xy, color='dimgray', transform=ccrs.PlateCarree(), zorder=1, alpha=0.8)
                listx = []
                listy = []
                for ind in range(peri_p.shape[0]):
                    listx.append(peri_p[ind][0])
                    listy.append(peri_p[ind][1])
                ax.plot(listx, listy)
        return perim