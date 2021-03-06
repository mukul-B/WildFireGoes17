import geopandas
import numpy as np
import pandas as pd

from keyvalues import viirs_file


def unique_time(date="2021-08-05", bbox=(-122.0, -119.0, 39.5, 41.5)):
    # bbox = (-122, 39.5, -119, 41.5)

    bbox2 = [bbox[0], bbox[2], bbox[1], bbox[3]]
    # print(bbox,bbox2)
    gdf = geopandas.read_file(viirs_file,bbox2)
    print(gdf.columns.tolist())
    # print(gdf["DAYNIGHT"])
    # 'INSTRUMENT': 'VIIRS'
    # n = gdf.SATELLITE
    gdf = gdf[gdf.ACQ_DATE == date]

    # uk = uk[uk.SATELLITE != '1']
    # uk = uk[uk.CONFIDENCE != 'l']
    gdf = gdf[gdf.DAYNIGHT == "D"]
    acq_time = gdf.ACQ_TIME
    nu = np.unique(acq_time).tolist()
    # nu = np.count_nonzero(gdf)
    return nu
    # if(nu):
    #     return [nu[0],nu[-1]]
    # else:
    #     return []
# ['0608', '0610', '0748', '0750', '0752', '0930', '0932', '1110', '1730', '1732', '1908', '1910', '1913', '2049', '2051', '2053']
# ['1730', '1732', '1908', '1910', '1913', '2049', '2051', '2053'] ( 10:30 - 2) , (11 - 4)
# 6880
# 4997899
# 103197
#  88785
#  14412
# ['0037', '1730', '1732', '1908', '1910', '1913', '1915', '1917', '2049', '2051', '2053', '2055', '2059', '2242']
# ['0605', '0608', '0610', '0746', '0748', '0750', '0752', '0928', '0930', '0932', '1104', '1108', '1110', '1246', '1429']

# ['0548', '0553', '0554', '0600', '0601', '0603', '0606', '0608', '0610', '0612', '0614', '0616', '0618', '0620', '0622', '0624', '0625', '0627', '0629', '0630', '0631', '0633', '0635', '0636', '0637', '0640', '0642', '0646', '0648', '0652', '0654', '0657', '0659', '0700', '0701', '0703', '0705', '0706', '0707', '0709', '0712', '0714', '0716', '0718', '0720', '0722', '0724', '0726', '0729', '0730', '0731', '0733', '0735', '0736', '0737', '0739', '0741', '0742', '0744', '0746', '0748', '0750', '0752', '0754', '0756', '0758', '0800', '0801', '0803', '0805', '0806', '0807', '0809', '0811', '0812', '0813', '0816', '0818', '0820', '0822', '0824', '0826', '0828', '0830', '0833', '0835', '0836', '0837', '0839', '0841', '0842', '0843', '0845', '0848', '0850', '0852', '0854', '0856', '0858', '0900', '0902', '0905', '0906', '0907', '0909', '0911', '0912', '0913', '0915', '0917', '0918', '0920', '0922', '0924', '0926', '0930', '0932', '0934', '0936', '0937', '0941', '0942', '0943', '0947', '0948', '0949', '0952', '0954', '0956', '0958', '1000', '1002', '1004', '1006', '1009', '1011', '1012', '1013', '1015', '1018', '1019', '1021', '1024', '1026', '1028', '1030', '1032', '1034', '1036', '1038', '1041', '1042', '1045', '1048', '1051', '1054', '1058', '1100', '1104', '1106', '1110', '1117']
# 11 pm -4
# ['1648', '1652', '1654', '1658', '1700', '1705', '1706', '1707', '1711', '1712', '1715', '1717', '1718', '1722', '1724', '1730', '1732', '1734', '1736', '1737', '1739', '1741', '1742', '1743', '1745', '1747', '1748', '1749', '1752', '1754', '1756', '1758', '1800', '1802', '1804', '1806', '1809', '1811', '1812', '1813', '1815', '1817', '1818', '1819', '1821', '1824', '1826', '1828', '1830', '1832', '1834', '1836', '1838', '1841', '1842', '1843', '1845', '1847', '1848', '1849', '1851', '1853', '1854', '1856', '1858', '1900', '1902', '1904', '1906', '1908', '1910', '1912', '1913', '1915', '1917', '1918', '1919', '1921', '1923', '1924', '1925', '1928', '1930', '1934', '1936', '1938', '1940', '1942', '1945', '1947', '1948', '1949', '1951', '1953', '1954', '1955', '1957', '2000', '2002', '2004', '2006', '2008', '2010', '2012', '2014', '2017', '2018', '2019', '2021', '2023', '2024', '2025', '2027', '2029', '2030', '2032', '2034', '2036', '2038', '2040', '2042', '2044', '2046', '2048', '2049', '2051', '2053', '2054', '2055', '2057', '2059', '2100', '2101', '2104', '2106', '2110', '2112', '2116', '2118', '2123', '2124', '2125', '2127', '2129', '2130', '2131', '2133', '2136', '2138', '2140', '2142', '2144', '2148', '2150', '2153', '2154', '2155', '2200', '2201', '2206', '2208', '2210']
# 10 am 3 pm
# VERSION ['1']
# SATELLITE ['N']
# INSTRUMENT ['VIIRS']

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

# print(unique_time("2021-08-06"))


# ['1942', '1948', '1954', '2000', '2006', '2012', '2018', '2024', '2030', '2036', '2042', '2048', '2054', '2100', '2106', '2112', '2118', '2124', '2130', '2136', '2142', '2148', '2154', '2200', '2206']
# ['0836', '0842', '0848', '0854', '0900', '0906', '0912', '0918', '0924', '0930', '0936', '0942', '0948', '0954', '1000', '1006', '1012', '1018', '1024', '1030', '1036', '1042', '1048', '1054', '1100', '1106', '1112']
