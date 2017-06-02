from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Operations():
    def __init__(self):
        pass
    
    def get_sparseNeighbors(self, dataIN, radius=500):
        '''
            input: A matrix/dataFrame (only numerical attributes)
            output: A Sparse matrix with cell value equal to 1 corresponsing to all nearest neighbors
            Function: Finds all the points (nearest neighbors) within a certain radius from each data point
        '''
        neigh = NearestNeighbors(radius=radius)
        neigh.fit(dataIN) 
        sparseNeighbors = neigh.radius_neighbors_graph(dataIN)
        return sparseNeighbors #sparseNeighbors.toarray()
    
    def standarize(self, dataIN):
        return (dataIN-np.mean(dataIN, axis=0))/np.std(dataIN, axis=0)

