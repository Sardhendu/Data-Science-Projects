
from main import dataBuilder, dataPrep, densityClusterBuilder, getCluster_Area

minDistance = 0.1
minSamples = 19
distanceMetric = 'euclidean'
how_many = 5
singleClusters = False
apha = 2.5

##################### Main Call
__main__ = True
if __main__:
	chicago_crm_pointsDir = '/Users/sam/All-Program/App-DataSet/Study/GeoSpatial-Analysis/Crimes2015_NA_rmv_sampl.csv'

	#### DATA BUILDER
	chicagoCrime = dataBuilder(chicago_crm_pointsDir)
	# print (chicagoCrime.head())

	#### DATA PREPARER
	dataUTM_scaled = dataPrep(chicagoCrime, sparseNeighbor=False)


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


	#### Get Each Cluster Area
	clusterArea = getCluster_Area(dataIN=dataUTM_scaled, 
							topClusterIndices_Dict=topClusterIndices_Dict,
							alpha = apha
						)

	cluster_groupByDF['ClusterArea'] = clusterArea.values()

	print (cluster_groupByDF)