import pandas as pd
import numpy as np

# Packages:
from SpatialHandlers import SpatialHandler
from Tools import Operations
from Clusters import DBSCAN_Cluters


#################### Initial Dataset Builder and Data Preparer

def dataBuilder(dataDIR):
	chicagoCrime = pd.read_csv(dataDIR)

	# Renaming Dataset:
	chicagoCrime = chicagoCrime[['Longitude', 'Latitude']]
	# chicagoCrime.head()
	# chicagoCrime.describe()

	# # Spatial Handler
	objSpHandler = SpatialHandler()
	objSpHandler.set_data(chicagoCrime)
	kwargs = {'to_UTM':True, 'to_GeoPoints':['UTM', 'LonLat']}
	(utmProj, geomPoints_UTM, geomPoints_LonLat) =  objSpHandler.transform_toUTM('Longitude', 'Latitude', **kwargs)

	# Now we add the columns to the DataFrame
	chicagoCrime['lonUTM'] = utmProj[:,0]
	chicagoCrime['latUTM'] = utmProj[:,1]
	chicagoCrime['geometryUTM'] = geomPoints_UTM
	chicagoCrime['geometryLonLat'] = geomPoints_LonLat

	# We would also like to add a colum with a different Mercetor projection
	objSpHandler.set_data(chicagoCrime)
	geomPointsMerc = objSpHandler.transform_toMerc('geometryLonLat', epsg=3857)
	chicagoCrime['geometryMerc'] = geomPointsMerc

	return chicagoCrime

# Prepare DataSet
def dataPrep(chicagoCrime, sparseNeighbor=False):
	dataOUT = np.array(chicagoCrime[['lonUTM','latUTM']])#[[0,0], [2,0], [0,2]]#[[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
	dataOUT = Operations().standarize(dataOUT)#(dataIN-np.mean(dataIN, axis=0))/np.std(dataIN, axis=0)
	print ('The shape of input data is: ', dataOUT.shape)

	# Calculating sparse neighbors, This preprocessing would help us to reduce 1 on 1 compare later with the density algorithm
	if sparseNeighbor:
		dataOUT = Operations().get_sparseNeighbors(dataOUT)

		# We would like to print the number of neighbors for some random samples
		# randNUM = np.random.randint(0,300,50)
		# print (np.sum(sparseNeighbors[randNUM].toarray(), axis=1))

	return dataOUT
	

###################### Clustering with DBSCAN
###################### Find Top Clusters (Individual Clusters)
def densityClusterBuilder(dataScaled, how_many = None):
	'''
		Input: 
			1. dataScaled: The Scaled Data.
			2. how_many: retrieve how many top clusters

		Output:
			1. clusterLabels : A list containing the label of each data set assigned to the cluster number
			2. cluster_groupByDF: A data frame consisting the cluster number and the count of elements in that cluster
			3. topClusterIndices_Dict : A dictionary containing the top most dense clusters and all the elements in it This output is only used for analysis, hence the default operation will not gather this data, unless specified by the user.

	'''
	## Clustering:
	objDBSCAN = DBSCAN_Cluters(eps=0.1, min_samples=19, metric='euclidean')
	objDBSCAN.set_data(dataScaled)
	clusterLabels = objDBSCAN.fit_predict()
	clusterUnqLabels = np.unique(clusterLabels)
	print ('The shape of cluster labels: ', clusterLabels.shape)
	print ('Unique Clusters Labels are: ', np.unique(clusterUnqLabels))

	## Analysis:
	cluster_groupByDF = objDBSCAN.cluster_info(clusterLabels)
	
	if how_many != None:
		if how_many > len(clusterUnqLabels):
			raise ValueError("Can't fulfil Request : Number of Top Cluster requested = %s, Number of clusters created = %s"%(str(how_many),str(len(clusterUnqLabels))))
		topClusterIndices_Dict = objDBSCAN.get_topClusters(clusterLabels=clusterLabels, how_many=how_many)
		return clusterLabels, cluster_groupByDF, topClusterIndices_Dict
	else:
		return clusterLabels, cluster_groupByDF


##################### Main Call
__main__ = True
if __main__:
	chicago_crm_pointsDir = '/Users/sam/All-Program/App-DataSet/Study/GeoSpatial-Analysis/Crimes2015_NA_rmv_sampl.csv'
	chicagoCrime = dataBuilder(chicago_crm_pointsDir)
	# print (chicagoCrime.head())
	dataUTM_scaled = dataPrep(chicagoCrime, sparseNeighbor=False)
	clusterLabels, cluster_groupByDF = densityClusterBuilder(dataUTM_scaled, how_many=None)
	print (cluster_groupByDF.head())

	# ## Add the cluster column to the dataframe:
	# chicagoCrimeNew = chicagoCrime[['lonUTM','latUTM']]
	# chicagoCrimeNew['clusterNo'] = clusterLabels

	# print (chicagoCrimeNew.head())


