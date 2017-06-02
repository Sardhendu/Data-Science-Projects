

from Plots import Plot
from ___main___ import dataBuilder, dataPrep, dataCluster

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
def clusterPlot(dataIN, clusterLabelsIN):
	obj_plot = Plot()
	obj_plot.set_figure(1,2, 600,1200)

	# Plot 1
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', plot_name='scatter LatLon'))
	obj_plot.base_plot(x=dataIN[:,0], y=dataIN[:,1])
	# Plot 2
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', marker_config=dict(color=clusterLabelsIN, colorscale='Viridis', showscale=True, size=6, opacity=0.6)))
	obj_plot.base_plot(x=dataIN[:,0], y=dataIN[:,1])
	obj_plot.show(plotFileName='Cluster-Plot')


def individualClusterPlot(dataIN, clusterLabels_IN):
	out = objDBSCAN.get_topClusters(clusterLabels=clusterLabels_IN, how_many=4)
	cluster_1 = dataIN.iloc[[values for key, values in out[0].items()][0],:]
	cluster_2 = dataIN.iloc[[values for key, values in out[1].items()][0],:]
	cluster_3 = dataIN.iloc[[values for key, values in out[2].items()][0],:]
	cluster_4 = dataIN.iloc[[values for key, values in out[3].items()][0],:]

	obj_plot = Plot()
	obj_plot.set_figure(2,2)

	# Plot 1:
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', plot_name='cluster1', marker_config=dict(size=4,opacity=0.3, color='black')))
	obj_plot.base_plot(x=dataIN['lonUTM'], y=dataIN['latUTM'])
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', marker_config=dict(size=4,opacity=0.3, color='blue')))
	obj_plot.add_plot(x=cluster_1['lonUTM'], y=cluster_1['latUTM'])

	#plot 2:
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', plot_name='cluster2', marker_config=dict(size=4,opacity=0.3, color='black')))
	obj_plot.base_plot(x=dataIN['lonUTM'], y=dataIN['latUTM'])
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', marker_config=dict(size=4,opacity=0.3, color='red')))
	obj_plot.add_plot(x=cluster_2['lonUTM'], y=cluster_2['latUTM'])

	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', plot_name='cluster3', marker_config=dict(size=4,opacity=0.3, color='black')))
	obj_plot.base_plot(x=dataIN['lonUTM'], y=dataIN['latUTM'])
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', marker_config=dict(size=4,opacity=0.3, color='orange')))
	obj_plot.add_plot(x=cluster_3['lonUTM'], y=cluster_3['latUTM'])

	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', plot_name='cluster4', marker_config=dict(size=4,opacity=0.3, color='black')))
	obj_plot.base_plot(x=dataIN['lonUTM'], y=dataIN['latUTM'])
	obj_plot.set_config(dict(plot_type='scatter', plot_mode='markers', marker_config=dict(size=4,opacity=0.3, color='green')))
	obj_plot.add_plot(x=cluster_4['lonUTM'], y=cluster_4['latUTM'])

	obj_plot.show()



density_plot = True
cluster_plot = True
individualClusterPlot = False

if density_plot:
	chicago_crm_pointsDir = '/Users/sam/All-Program/App-DataSet/Study/GeoSpatial-Analysis/Crimes2015_NA_rmv_sampl.csv'
	chicagoCrime = dataBuilder(chicago_crm_pointsDir)
	densityPlot(chicagoCrime)
if cluster_plot:
	# print (chicagoCrime.head())
	dataUTM_scaled = dataPrep(chicagoCrime, sparseNeighbor=False)
	clusters, cluster_groupByDF = dataCluster(dataUTM_scaled)
	clusterPlot(clusters)
if individualClusterPlot:
	individualClusterPlot(chicagoCrimeNew, clusterLabels_IN)
