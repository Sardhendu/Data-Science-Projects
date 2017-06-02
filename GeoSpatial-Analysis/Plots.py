from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from copy import deepcopy
import numpy as np
import pandas as pd


# Plotting Libraries
import plotly
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
# plotly.offline.init_notebook_mode() 
plotly.offline.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
plt.style.use('ggplot')




class GlobalPlot(object):
    def __init__(self):
        pass
        
    def get_Plot_info(self):
        plotInfo = dict(
            plot_type = 'scatter, histogram2Dcontours',
            plot_mode = "lines, markers, lines+markers, lines+markers+text",
            marker_conf = dict(
                size = '3 to 10',
                opacity = '0.3 to 0.7',
                color = 'black, white , etc. When using group on a third column then pass in the column values;',
                colorscale ='Viridis etc. Use when performing a group by on a third column values. Optimal when many values in 3rd column',
                showscale = 'True or False : Should you choose to activate the color scale or not'
            )
        ) 
        
        return plotInfo
        
    def get_GeoPlot_info(self):
        pass
    
    def set_figure(self):
        pass
    
    def base_plot(self):
        pass
    
    def add_plot(self):
        pass


class GeoPlot():
    def __init__(self):
        GlobalPlot.__init__(self)
        self.newPlot = None
        self.basePlot = None
        self.axPointer = None
        
        # We basically make a default subplot, However this can be overridden by calling
        # the set_figure function
        self.fig, self.ax = plt.subplots(1,1, 
                                         figsize=(10, 10), 
                                         facecolor='w', edgecolor='k')
        
    def set_figure(self, numRows=1, numColumns=1, lenXaxis=40, lenYaxis=15):
        self.fig, self.ax = plt.subplots(numRows, 
                                         numColumns, 
                                         figsize=(lenXaxis, lenYaxis), 
                                         facecolor='w', edgecolor='k')
        if numRows>1 or numColumns>1:
            self.ax = self.ax.ravel()
            self.axPointer = 0
            
            
    def set_data(self, dataIN):
        self.dataIN = dataIN
        
    def base_plot(self, color='white', dataIN=[]):
        if any(dataIN):
            self.dataIN = dataIN
            
        if self.axPointer != None:
            self.basePlot = self.dataIN.plot(ax=self.ax[self.axPointer], color=color)
            self.axPointer += 1
        else:
            self.basePlot = self.dataIN.plot(ax=self.ax, color=color)
            
    def add_plot(self, dataIN):
        if self.newPlot == None:
            self.newPlot = dataIN.plot(ax=self.basePlot, marker='o', color='red', markersize=7)
        else:
            self.newPlot = dataIN.plot(ax=self.newPlot, marker='o', color='red', markersize=7)       



class Plot(GlobalPlot):
    def __init__(self):
        GlobalPlot.__init__(self)
        self.fig = tools.make_subplots(rows=1, cols=1)
        self.fig['layout'].update(height=400, width=400)
        
        # Plot Layout/config default
        self.plot_type = 'scatter'
        self.plot_mode = 'lines+markers'
        self.plot_name = 'noName_given'
        self.marker_config=dict(
                        size='6',
                        opacity=0.3,
                        color = 'black',
                    )
        
    def set_config(self, conf=None):
        if conf:
            confKeys = conf.keys()
        else:
            confKeys = []
            
        if 'plot_type' in confKeys:
            self.plot_type = conf['plot_type']
            
        if 'plot_mode' in confKeys:
            self.plot_mode = conf['plot_mode']            
            
        if 'plot_name' in confKeys:
            self.plot_name = conf['plot_name'] 
        
        if 'marker_config' in confKeys:
            self.marker_config = conf['marker_config']
            
            
    def get_plot(self):
        if self.plot_type=='scatter':
            if self.plot_mode == 'markers':
                trace = go.Scatter(
                        x=self.x,
                        y=self.y,
                        name = self.plot_name,
                        mode=self.plot_mode,
                        marker=self.marker_config#Marker(color='black', size=3, opacity=0.2)
                    )
            else:
                trace = go.Scatter(
                        x=self.x,
                        y=self.y,
                        name = self.plot_name,
                        mode=self.plot_mode
                    )
                
        elif self.plot_type=='hist2Dcontour':
            trace = go.Histogram2dcontour(
                        x=self.x,
                        y=self.y,
                        name = self.plot_name,
                        ncontours=20,
                        colorscale='Hot',
                        reversescale=True, 
                        showscale=False
                    )
        else:
            raise Exception('You should specify at one type of plot')
            
        return trace
        
        
    def set_figure(self, numRows, numColumns, height=None, width=None, title=None):
        self.numRows = numRows
        self.numColumns = numColumns
        if height:
            height = height
        else:
            height = numRows*600
            
        if width:
            width = width
        else:
            width = numColumns*600
        
        
        self.fig = tools.make_subplots(rows=numRows, cols=numColumns)
        if title:
            self.fig['layout'].update(height=height, width=width, title=title)
        else:
            self.fig['layout'].update(height=height, width=width)
        
        self.subplotList = list(itertools.product(np.arange(numRows)+1, np.arange(numColumns)+1))
        self.subplotIndex = 0
        
    def set_data(self, dataIN):
        self.x = dataIN.iloc[:,0]
        self.y = dataIN.iloc[:,1]
        
    def base_plot(self, x=False, y=False):#dataIN=False):
        '''  
            Give as input a DataFrame with two columns as input
            Where the 1st column is to be plotted in the X axis
            and the 2nd column is to be plotted in y axis
        '''
        (self.rowIndex,  self.columnIndex) = self.subplotList[self.subplotIndex]
        
        if any(x):
            self.x = x
        if any(y):
            self.y = y

        trace = self.get_plot()
        self.fig.append_trace(trace, self.rowIndex, self.columnIndex)
        self.subplotIndex += 1
        
    def add_plot(self, x=False, y=False):   
        # When we add_plot we actually overlay the new plot on top of the provious plot.
        # Since we increase the base_plot every time. We decrease it by 1 here so that we
        # can overlay the new plot on the base plot
        self.subplotIndex -= 1
        (self.rowIndex,  self.columnIndex) = self.subplotList[self.subplotIndex]
        
        if any(x):
            self.x = x
        if any(y):
            self.y = y
            
        trace = self.get_plot()
        self.fig.append_trace(trace, self.rowIndex, self.columnIndex)
        self.subplotIndex += 1
        
    def show(self, plotFileName = 'Temp-Plot'):
        plot(self.fig, filename=plotFileName)




