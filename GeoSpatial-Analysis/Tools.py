from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay, distance
import shapely.geometry as geometry
from sklearn import metrics

import pandas as pd
import numpy as np
import math

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


class polygonArea():

    def __init__(self):
        self.edges =  set()
        self.edge_points = []
        
    def add_edge(self, coords, i, j):
            """
                Add an edge between two near vertices if not already exists
            """
            if (i, j) in self.edges or (j, i) in self.edges:
                return
            self.edges.add((i, j))
            self.edge_points.append(coords[ [i, j] ])

#     def convex_hull():
        
    def alpha_shape(self, points, alpha):
        """
            Compute the alpha shape (concave hull) of a set
            of points.

            Inputs: Points - The input 2D points
                    alpha  - A sort of threshold to trade off between concave Hull and Convex Hull
                            The more the value of alpha the more less the area of the polygon connecting points (high order conccave hull)
                            The less the value of alpha the less the area of the polygon connection points (Convex hull)
            Output: polygonPoints - The polygon points (outer most points) extracted based on the input alpha
        """
        if len(points) < 4:
            # When you have a triangle, there is no sense
            # in computing an alpha shape.
            return geometry.MultiPoint(list(points)).convex_hull

        # Find all the non overlapping triangle
        coords = np.array(points)
        tri = Delaunay(coords)

        for ia, ib, ic in tri.vertices:
            # Calculate the Lengths of sides of triangle (This is simply the euclidean distance)
            a = metrics.pairwise.euclidean_distances(coords[ia].reshape(1, -1) , coords[ib].reshape(1, -1))[0][0]
            b = metrics.pairwise.euclidean_distances(coords[ib].reshape(1, -1) , coords[ic].reshape(1, -1))[0][0]
            c = metrics.pairwise.euclidean_distances(coords[ic].reshape(1, -1) , coords[ia].reshape(1, -1))[0][0]

            # Find Area of triangle by Heron's formula
            # Semiperimeter of triangle, half of teh total perimeter
            s = (a + b + c)/2.0
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)

            # The more the distance between points a,b,c the more will be the circum_r
            # This step will basically avoid connecting far edges, which are mostly 
            # a result of convex hull. In other words if the points are near to each other 
            # till a certain threshold only then the edges are retained between each vertices.
            if circum_r < 1.0/alpha:
                self.add_edge(coords, ia, ib)
                self.add_edge(coords, ib, ic)
                self.add_edge(coords, ic, ia)
    
        # edge_points are the points that qualifies the given alpha value. A Edge point 
        m = geometry.MultiLineString(self.edge_points)
        triangles = list(polygonize(m))

        # Cascade_union will remove all the tringle edge that are not at the exterior
        # and only keep the once that are the outer most edges as a result of the input alpha.
        # Hence creating a single polygon which makes it easier to find the area and makes it easier to plot.
        polygonPoints = cascaded_union(triangles)

        return polygonPoints.area



