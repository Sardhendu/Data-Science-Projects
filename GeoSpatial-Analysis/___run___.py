import random
import numpy as np
import pandas as pd
import DataGenerator
from main import dataCleaner, dataBuilder, dataPrep, densityClusterBuilder, getCluster_Area


# topClusterIndicesDict_dir =


minDistance = 0.1
minSamples = 19
distanceMetric = 'euclidean'
how_many = 4
singleClusters = False
alpha = 2.5

deviceID = None
indexNum=99
num_rand_data=1


__main__ = True
__analysis__ = False

##################### Main Call

if __main__:
	#### Generate Data
	deviceLocationData = DataGenerator.generateData(indexNum=indexNum)

    #### DATA CLEANER
	cleanData = dataCleaner(deviceLocationData).reset_index()
	print (cleanData.head())

    #### DATA BUILDER
	spatialData = dataBuilder(cleanData)
	print (spatialData.head())

    #### DATA PREPARER  Returns a numpy array
	dataUTM_scaled = dataPrep(spatialData, sparseNeighbor=False)
	print(dataUTM_scaled)

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
	# Storing topClusterIndices_Dict into data store


	print (type(topClusterIndices_Dict))
	# Select only the top clusters
	cluster_groupByDF = cluster_groupByDF.iloc[0:how_many, :]
	print(cluster_groupByDF)

	# #### Get Each Cluster Area
	#  clusterArea = getCluster_Area(dataIN=dataUTM_scaled,
	#  						topClusterIndices_Dict=topClusterIndices_Dict,
	#  						alpha = apha
	#  					)

    # cluster_groupByDF['ClusterArea'] = clusterArea.values()

    # print (cluster_groupByDF)



if __analysis__:
	locationData = DataGenerator.generateData(indexNum=indexNum)
	print(locationData)

	deviceAddrData = DataGenerator.generateTruthData(indexNum=indexNum)
	print(deviceAddrData)