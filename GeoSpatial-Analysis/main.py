import pandas as pd
import numpy as np
from collections import OrderedDict

# Packages:
from SpatialHandlers import SpatialHandler
from Tools import Operations, polygonArea
from Clusters import DBSCAN_Cluters


#################### Data Cleaner

def dataCleaner(dataIN):
    # Quickly loop through the entire dataset and remove bad entries.
    '''
    	Input: A data frame with atleast two columns
    		1> Latitude and
    		2> Longitude
        Output: A data frame with
        	1> Every input column but clean rows
        	2> A new index column that is the row number of the actual input data set

        1. Remove entries with latitue or longitude values as 0
        2. Remove entries with latitude or longitude that have 3 or less than three decimal points, because they will not add much knowledge to pin pointing the location

        Note: To Confirm
        42.737000 is treated as 42.737 and since 42.737 has only three digits after the decimal points, this is removed form the list
    '''
    # dataIN_copy = dataIN.copy(deep=True)
    dataIN = dataIN.reset_index()
    badDataIndices = [num for num, (lat, lon) in enumerate(zip(dataIN['Latitude'], dataIN['Longitude'])) if
                      lat == 0 or lon == 0 or str(lat)[::-1].find('.') <= 3 or str(lon)[::-1].find('.') <= 3]

    # print (len(dataIN))
    print(badDataIndices)

    if len(badDataIndices) != len(set(badDataIndices)):
        raise ValueError('Few Indices repeat multiple times')

    cleanedData = dataIN.loc[~dataIN.index.isin(badDataIndices)]

    if len(cleanedData) + len(badDataIndices) != len(dataIN):
        print('')
        print(len(dataIN))
        print(len(cleanedData))
        print(len(badDataIndices))
        raise ValueError('The length of badData and Clean data should equal the lenght of total input data')

    return cleanedData


#################### Initial Dataset Builder and Data Preparer

def dataBuilder(dataIN):
	'''
		Input: A data frame with atleast two columns
			1> Latitude and
			2> Longitude
		Output: A data frame with
			1> Every input column including the index column
			2> If specified 'lonUTM' and 'latUTM
			3> If specified Geometric Lat Lon points
			4> If specified Geometric Mercetor points
	'''

    # # Spatial Handler
    objSpHandler = SpatialHandler()
    objSpHandler.set_data(dataIN)
    kwargs = {'to_UTM': True}#, 'to_GeoPoints': ['UTM', 'LonLat']}
    (utmProj, geomPoints_UTM, geomPoints_LonLat) = objSpHandler.transform_toUTM('Longitude', 'Latitude', **kwargs)

    # Now we add the columns to the DataFrame
    dataIN['lonUTM'] = utmProj[:, 0]
    dataIN['latUTM'] = utmProj[:, 1]
    # dataIN['geometryUTM'] = geomPoints_UTM
    # dataIN['geometryLonLat'] = geomPoints_LonLat

    ############
    # The below produces error because some of the latitude and longitude points becomes high degree of floating point such as POINT (-84.40524000000001 42.770348). Hence we have to handle this scenario
    ############
    # We would also like to add a colum with a different Mercetor projection
    # objSpHandler.set_data(dataIN)
    # geomPointsMerc = objSpHandler.transform_toMerc(column_LonLat='geometryLonLat',
    # 												dataIN=dataIN,
    # 												epsg=3857
    # 											)
    # dataIN['geometryMerc'] = geomPointsMerc

    return dataIN


# Prepare DataSet
def dataPrep(dataIN, sparseNeighbor=False):
    dataOUT = np.array(dataIN[['lonUTM', 'latUTM']])  # [[0,0], [2,0], [0,2]]#[[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    dataOUT = Operations().standarize(dataOUT)  # (dataIN-np.mean(dataIN, axis=0))/np.std(dataIN, axis=0)
    print('The shape of input data is: ', dataOUT.shape)

    # Calculating sparse neighbors, This preprocessing would help us to reduce 1 on 1 compare later with the density algorithm
    if sparseNeighbor:
        dataOUT = Operations().get_sparseNeighbors(dataOUT)

    # We would like to print the number of neighbors for some random samples
    # randNUM = np.random.randint(0,300,50)
    # print (np.sum(sparseNeighbors[randNUM].toarray(), axis=1))

    return dataOUT


###################### Clustering with DBSCAN
###################### Find Top Clusters (Individual Clusters)
def densityClusterBuilder(dataIN, eps, minSamples, distanceMetric='euclidean', how_many=None, singleClusters=False):
    '''
        Input:
            1. dataIN: The Scaled Data.
            2. how_many: retrieve how many top clusters
        Output:
            1. clusterLabels : A list containing the label of each data set assigned to the cluster number
            2. cluster_groupByDF: A data frame consisting the cluster number and the count of elements in that cluster
            3. topClusterIndices_Dict : A dictionary containing the top most dense clusters and all the elements in it This output is only used for analysis, hence the default operation will not gather this data, unless specified by the user.
    '''
    ## Clustering:
    objDBSCAN = DBSCAN_Cluters(eps=eps, min_samples=minSamples, metric=distanceMetric)
    objDBSCAN.set_data(dataIN)
    clusterLabels = objDBSCAN.fit_predict()
    clusterUnqLabels = np.unique(clusterLabels)
    print('The shape of cluster labels: ', clusterLabels.shape)
    print('Unique Clusters Labels are: ', np.unique(clusterUnqLabels))

    ## Analysis:
    cluster_groupByDF = objDBSCAN.cluster_info(clusterLabels)

    if how_many != None:
        if how_many > len(clusterUnqLabels):
            raise ValueError(
                "Can't fulfil Request : Number of Top Cluster requested = %s, Number of clusters created = %s" % (
                str(how_many), str(len(clusterUnqLabels))))
        topClusterIndices_Dict = objDBSCAN.get_topClusters(clusterLabels=clusterLabels, how_many=how_many,
                                                           singleClusters=singleClusters)
        return clusterLabels, cluster_groupByDF, topClusterIndices_Dict
    else:
        return clusterLabels, cluster_groupByDF


def getCluster_Area(dataIN, topClusterIndices_Dict, alpha):
    clusterArea = OrderedDict()
    for key, values in topClusterIndices_Dict.items():
        clusterArea[key] = polygonArea().alpha_shape(dataIN[values, :], alpha=alpha)
    return clusterArea