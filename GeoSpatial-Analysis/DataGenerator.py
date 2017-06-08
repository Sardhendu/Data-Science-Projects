from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import random

random.seed(78672)

mac = True

if mac :
	UM_LatLon_dir = '/Users/sam/All-Program/App-DataSet/Data-Science-Projects/Geo-Spatial-Analysis/UM_transactions_devices.csv'
	Device_LatLon_dir = '/Users/sam/All-Program/App-DataSet/Data-Science-Projects/Geo-Spatial-Analysis/UM_devices_address_latlong.csv'
else:
	UM_LatLon_dir = 'C:\\Users\\swmishr\\Documents\\All-Program\\App-Dataset\\UM_transactions_devices.csv'
	Device_LatLon_dir = "C:\\Users\\swmishr\\Documents\\All-Program\\App-Dataset\\UM_devices_address_latlong.csv"


#40, 199

def generateData(deviceID=None, indexNum=None, num_rand_data=1):
	locationData = pd.read_csv(UM_LatLon_dir, header=None)
	locationData.columns = ['deviceID', 'Latitude', 'Longitude', 'timeStamp']

	unqdeviceID = np.unique(locationData['deviceID'])

	if deviceID != None:
		deviceLocationData = locationData.loc[locationData['deviceID'] == deviceID]
		  # 199
	elif indexNum != None:
		deviceLocationData = locationData.loc[locationData['deviceID'] == unqdeviceID[indexNum]]
	else:
		random.shuffle(unqdeviceID)
		deviceLocationData = locationData.loc[locationData['deviceID'] == unqdeviceID[0]]

	return deviceLocationData




def generateTruthData(deviceID=None, indexNum=None, num_rand_data=1):
	locationData = pd.read_csv(Device_LatLon_dir, header=None)
	locationData.columns = ['deviceID', 'TU_KEY', 'STREET_NBR', 'PRE_DIR', 'STREET_NME', 'STREET_TYP', 'POST_DIR', 'CITY', 'STATE', 'ZIP_CODE', 'Latitude', 'Longitude', 'LAT_LONG_MATCH_LEVEL']

	unqdeviceID = np.unique(locationData['deviceID'])

	if deviceID != None:
		deviceLocationData = locationData.loc[locationData['deviceID'] == deviceID]
	# 199
	elif indexNum != None:
		deviceLocationData = locationData.loc[locationData['deviceID'] == unqdeviceID[indexNum]]
	else:

		random.shuffle(unqdeviceID)
		deviceLocationData = locationData.loc[locationData['deviceID'] == unqdeviceID[0]]

	return deviceLocationData[['deviceID','Latitude','Longitude']]


