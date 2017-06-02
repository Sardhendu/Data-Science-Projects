from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import itertools
from copy import deepcopy
import numpy as np
import pandas as pd


# Spatial Libraries
import utm
import shapely as shp
import geopandas as gpd
from shapely.geometry import Point

class SpatialHandler():
    
    def __init__(self):
        pass
    
    def set_data(self, dataIN):
        self.dataIN = dataIN
          
    def latlon_toUTM(self, lon, lat):
        '''
            Input: Latitude and Longitude
            Output: Longitude and Latitude UTM projections 
        '''
        tup = utm.from_latlon(lat,lon)
        lat_cnvrt = tup[0]
        lon_cnvrt = tup[1]

        return lon_cnvrt, lat_cnvrt
    
    def transform_toUTM(self, column_Lon, column_Lat, **kwargs):
        utmProj = []
        geomPoints_UTM = []
        geomPoints_LonLat = []
        
        kwargsKeys = kwargs.keys()
        for lon, lat in zip(self.dataIN[column_Lon], self.dataIN[column_Lat]):
            if 'to_UTM' in kwargsKeys:
                columnOut_1 = self.latlon_toUTM(lon, lat)
                utmProj.append(columnOut_1) 
                
            if 'to_GeoPoints' in kwargsKeys and 'UTM' in kwargs['to_GeoPoints']:
                try:
                    geomPoints_UTM.append(Point(columnOut_1))   
                except UnboundLocalError:
                    raise UnboundLocalError('You should first transform the data into UTM !!')
                    
            
            if 'to_GeoPoints' in kwargsKeys and 'LonLat' in kwargs['to_GeoPoints']:
                geomPoints_LonLat.append(Point(lon,lat))
                
                
        return (np.array(utmProj, dtype='float64'), 
                gpd.GeoSeries(geomPoints_UTM), 
                gpd.GeoSeries(geomPoints_LonLat))
    
    
    def transform_toMerc(self, column_LonLat, dataIN=[], epsg=3395):
        if any(dataIN):
            self.dataIN = dataIN
            
        self.dataIN = self.dataIN.set_geometry(column_LonLat)
        
        LonLat = self.dataIN[column_LonLat]
        LonLat.crs = {'init': 'epsg:4326'}       ## Initialize the current column as espg:4326 for further conversion
        epsgStr = 'epsg:'+str(epsg)
        geomPoints_Merc = LonLat.to_crs({'init': epsgStr})  ## Create a new GeoSeries with Mercector projection
        
        return geomPoints_Merc
