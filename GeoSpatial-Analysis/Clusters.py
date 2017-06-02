from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import itertools
from copy import deepcopy
import numpy as np
import pandas as pd


# Clustering Libraries
from sklearn.cluster import DBSCAN



class DBSCAN_Cluters():
    def __init__(self, eps=0.5,             # Maximum distance between two datapoints
                 min_samples=5,             # number of minimum neighbors around the point to be considered as a core point
                 metric='euclidean',        # The distance metric to be used   
                 algorithm='auto',          # The datastructure to be used while making search example: ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
                 leaf_size=30,              # Number of leaves node
                 p=None,                    # The power of miskowski metric 1 for L1(norm), 2 for L2(euclidean norm)
                 n_jobs=1):                 # If -1 then the number of jobs are considered equal to the number of CPU cores
        '''
             eps -> Maximum distance between two points to be considered in the same neighborhood
             min_samples -> The number of samples in a neighborhood for a point to be considered as a core point.
        '''
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, 
                             metric=metric, algorithm=algorithm, 
                             leaf_size=leaf_size, p=p,
                             n_jobs=n_jobs)
    
    def set_data(self, dataIN):
        self.dataIN = dataIN

    def fit(self):
        fitObj = self.dbscan.fit(deepcopy(self.dataIN))
        
    def fit_predict(self):
        '''
            Fits the DBSCAN model and transforms the data 
            (For unsupervised algorithm we apply the predict on the data)
        '''
        clusterLabels = self.dbscan.fit_predict(deepcopy(self.dataIN))
        return clusterLabels
    
    def cluster_info(self, clusterLabels):
        clst = pd.DataFrame({'clusterNo':clusterLabels, 'count':np.ones(len(clusterLabels), dtype=int)})
        groupByDF = (clst.groupby('clusterNo')
           .agg({'count':'sum'})
           .sort_values('count', ascending=False)
           .reset_index())
        return groupByDF
    
    def get_topClusters(self, clusterLabels, cluster_groupByDF=[], how_many=1, single_clusters=False):
        if not any(cluster_groupByDF):
            cluster_groupByDF = self.cluster_info(clusterLabels)
        
        if not single_clusters:
            cluster_groupByDF = cluster_groupByDF.loc[cluster_groupByDF['clusterNo'] != -1]
            
        clusterNo = np.array(cluster_groupByDF['clusterNo'])
        
        topClusterIndices_Dict = {}
        for num, clstr_no in enumerate(clusterNo):
            if num == how_many:
                break
            indices = np.where(clusterLabels == clstr_no)[0]
            topClusterIndices_Dict[num] = {str(clstr_no):indices}
    
        return topClusterIndices_Dict
    # Now we fit the data and cluster the data.
