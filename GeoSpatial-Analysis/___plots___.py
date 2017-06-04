
import numpy as np
from Plots import Plot
from ___main___ import dataBuilder, dataPrep, densityClusterBuilder

################## Density plots

def densityPlot(datIN):
	obj_plot = Plot()
	obj_plot.set_figure(2,2, 1200,1500)

	# plot1
	obj_plot.set_config({'plot_type':'scatter', 'plot_mode':'markers', 'plot_name':'scatter_LonLat'})
	obj_plot.base_plot(x=datIN['Longitude'], y=datIN['Latitude'])

	# plot2
	obj_plot.set_config({'plot_type':'scatter', 'plot_mode':'markers', 'plot_name':'scatter_UTM'})
	obj_plot.base_plot(x=datIN['lonUTM'], y=datIN['latUTM'])

	# plot3
	obj_plot.set_config({'plot_type':'scatter', 'plot_mode':'markers', 'plot_name':'histogram2Dcontours-LonLat'})
	obj_plot.base_plot(x=datIN['Longitude'], y=datIN['Latitude'])
	obj_plot.set_config({'plot_type':'hist2Dcontour'})
	obj_plot.add_plot(x=datIN['Longitude'], y=datIN['Latitude'])

	# plot4
	obj_plot.set_config({'plot_type':'scatter', 'plot_mode':'markers', 'plot_name':'histogram2Dcontours-UTM'})
	obj_plot.base_plot(x=datIN['lonUTM'], y=datIN['latUTM'])
	obj_plot.set_config({'plot_type':'hist2Dcontour'})
	obj_plot.add_plot(x=datIN['lonUTM'], y=datIN['latUTM'])
	obj_plot.show(plotFileName='Histogram-2D-contour-Plot')




#################### Cluster Plot
## Plots:
def densityClusterPlot(dataIN, clusterLabelsIN):
	obj_plot = Plot()
	obj_plot.set_figure(1,2, 600,1200)

	# Plot 1
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', plot_name='scatter LatLon'))
	obj_plot.base_plot(x=dataIN[:,0], y=dataIN[:,1])
	# Plot 2
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', marker_config=dict(color=clusterLabelsIN, colorscale='Viridis', showscale=True, size=6, opacity=0.6)))
	obj_plot.base_plot(x=dataIN[:,0], y=dataIN[:,1])
	obj_plot.show(plotFileName='Cluster-Plot')


def topClusterPlot(dataIN, list_of_topClustersDF, how_many=4):#dataIN, clusterLabels_IN):
	obj_plot = Plot()
	obj_plot.set_figure(int(np.ceil(how_many/2)), 2)

	for clusterNum, clusterPoints in enumerate(list_of_topClustersDF):
			# Plot 1:
		obj_plot.set_config(dict(plot_type='scatter', 
			plot_mode='markers', 
			plot_name='cluster'+str(clusterNum), 
			marker_config=dict(size=4,opacity=0.3, color='black'))
		)
		obj_plot.base_plot(x=dataIN['lonUTM'], y=dataIN['latUTM'])

		obj_plot.set_config(dict(plot_type='scatter', 
			plot_mode='markers', 
			marker_config=dict(size=4,opacity=0.3, color='blue'))
		)
		obj_plot.add_plot(x=clusterPoints['lonUTM'], y=clusterPoints['latUTM'])

	obj_plot.show(plotFileName='Top Clusters - Plot')



density_plot = True
cluster_plot = True
top_cluster_plot = True
how_many_top_clusters = 6

if density_plot:
	chicago_crm_pointsDir = '/Users/sam/All-Program/App-DataSet/Study/GeoSpatial-Analysis/Crimes2015_NA_rmv_sampl.csv'
	chicagoCrime = dataBuilder(chicago_crm_pointsDir)
	densityPlot(chicagoCrime)

if density_plot and cluster_plot:
	# print (chicagoCrime.head())
	dataUTM_scaled = dataPrep(chicagoCrime, sparseNeighbor=False)
	clusterLabels, cluster_groupByDF, topClusterIndices_Dict = densityClusterBuilder(dataUTM_scaled, how_many=how_many_top_clusters)
	print (clusterLabels)

	# Plot the cluster density plot
	densityClusterPlot(dataUTM_scaled, clusterLabels)

	## Add the cluster column to the dataframe:
	chicagoCrimeNew = chicagoCrime[['lonUTM','latUTM']]
	chicagoCrimeNew['clusterNo'] = clusterLabels

	if top_cluster_plot:
		list_of_topClustersDF = []

		for num in np.arange(how_many_top_clusters): 
			list_of_topClustersDF.append(chicagoCrimeNew.iloc[[values for key, values in topClusterIndices_Dict[num].items()][0],:])
		topClusterPlot(chicagoCrimeNew, list_of_topClustersDF, how_many=how_many_top_clusters)
