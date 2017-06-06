
import random
import numpy as np
import pandas as pd
from main import dataCleaner, dataBuilder, dataPrep, densityClusterBuilder, getCluster_Area

minDistance = 0.1
minSamples = 19
distanceMetric = 'euclidean'
how_many = 5
singleClusters = False
apha = 2.5


random.seed(78672)

chicago_crm_pointsDir = '/Users/sam/All-Program/App-DataSet/Study/GeoSpatial-Analysis/Crimes2015_NA_rmv_sampl.csv'

UM_LatLon_dir = '/Users/sam/All-Program/App-DataSet/Data-Science-Projects/Geo-Spatial-Analysis/UM_transactions_devices.csv'

locationData = pd.read_csv(UM_LatLon_dir,  header=None)
locationData.columns = ['deviceID', 'Latitude', 'Longitude', 'timeStanp']

unqdeviceID = np.unique(locationData['deviceID'])
random.shuffle(unqdeviceID)


locationData = locationData.loc[locationData['deviceID'] == unqdeviceID[199]]


##################### Main Call
__main__ = True
if __main__:
	
	#### DATA CLEANER
	cleanData = dataCleaner(locationData).reset_index()

	#### DATA BUILDER
	spatialData = dataBuilder(cleanData)

	# #### DATA PREPARER  Returns a numpy array
	dataUTM_scaled = dataPrep(spatialData, sparseNeighbor=False)
	print (dataUTM_scaled)

	#### DENSITY CLUSTERING
	(clusterLabels, 
		cluster_groupByDF, 
		topClusterIndices_Dict) = densityClusterBuilder(dataIN=dataUTM_scaled, 
											eps=minDistance,
											minSamples=minSamples,
											distanceMetric=distanceMetric,
											how_many=how_many, 
											singleClusters=singleClusters
										)
	# Select only the top clusters
	cluster_groupByDF = cluster_groupByDF.iloc[0:how_many,:]
	print (cluster_groupByDF)

	# #### Get Each Cluster Area
	# clusterArea = getCluster_Area(dataIN=dataUTM_scaled, 
	# 						topClusterIndices_Dict=topClusterIndices_Dict,
	# 						alpha = apha
	# 					)

	# cluster_groupByDF['ClusterArea'] = clusterArea.values()

	# print (cluster_groupByDF)