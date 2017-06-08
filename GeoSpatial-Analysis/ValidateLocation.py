from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd


truthData_dir = "/Users/sam/All-Program/App-DataSet/Data-Science-Projects/Geo-Spatial-Analysis/UM_devices_address_latlong.csv"

truthData = pd.read_csv(truthData_dir, header=None)

# print (truthData.head())


truthData.columns = ['DEVICE_ID', 'TU_KEY', 'STREET_NBR', 'PRE_DIR', 'STREET_NME', 'STREET_TYP', 'POST_DIR', 'CITY', 'STATE', 'ZIP_CODE', 'LATITUDE', 'LONGITUDE', 'LAT_LONG_MATCH_LEVEL']



