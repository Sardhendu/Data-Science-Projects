import numpy as np
import pandas as pd
import sklearn as sk
import utm
# import geopandas as gpd
import shapely as shp
import plotly

mac = False
if mac :
    UM_LatLon_dir = '/Users/sam/All-Program/App-DataSet/Data-Science-Projects/Geo-Spatial-Analysis/UM_devices_address_latlong.csv.csv'
else:
    UM_LatLon_dir = 'C:\\Users\\swmishr\\Documents\\All-Program\\App-Dataset\\UM_devices_address_latlong.csv'

dataIN = pd.read_csv(UM_LatLon_dir, header=None)

print (dataIN.head())
