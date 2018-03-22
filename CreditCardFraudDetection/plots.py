import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
sns.set_style("whitegrid")


class GlobalPlot(object):
    def __init__(self, rows, columns, fig_size):
        fig_size = tuple(fig_size)
        f, axs = plt.subplots(rows, columns, figsize=fig_size)
        self.axs_ind = 0
    
        if rows == 1 and columns == 1:
            self.axs = [axs]
        else:
            self.axs = axs.ravel()
            
    def help(self):
        help_string = ''
        help_string = help_string + 'ROC_CURVE: Plot().vizualize(data=pd.DataFrame({"fpr":fpr,"tpr":tpr}), ' \
                                    'colX="fpr", colY="tpr", label_col=None, viz_type="roc", params={"title":"your_title"})   '
        help_string = help_string + 'LINE_PLOT: Plot().vizualize(data=pd.DataFrame(your_array, ' \
                                  'columns=["your_column_name"]), ' \
                      'colX=None, colY=None, label_col=None, viz_type="line",  params={"title":"your_title"})   '
        
        return help_string
            
            
class Plot(GlobalPlot):
    def __init__(self, rows=1, columns=1, fig_size=[8, 8]):
        GlobalPlot.__init__(self, rows, columns, fig_size)
    
    def vizualize(self, data, colX, colY=None, label_col=None, viz_type='bar', params={}):
        '''
            params : title,
            data should be a data frame
        '''
        params_keys = list(params.keys())
        if viz_type == 'dist':
            data = data.reset_index().drop('index', 1)
            sns.distplot(data, ax=self.axs[self.axs_ind])
            if 'title' in params_keys:
                self.axs[self.axs_ind].set_title(params['title'])
                
        if viz_type == 'hist':
            ax = self.axs[self.axs_ind]
            if 'bins' in params_keys:
                ax.hist(data[colX], bins=params['bins'])
            else:
                ax.hist(data[colX])
            
            if 'xlabel' in params_keys and 'ylabel' in params_keys:
                ax.set(xlabel=params['xlabel'], ylabel=params['ylabel'])
                
            if 'title' in params_keys:
                ax.set_title(params['title'])
                
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
        
        if viz_type == 'scatter':
            data.plot.scatter(x=colX, y=colY, c=label_col, colormap='Dark2', ax=self.axs[self.axs_ind])
        
        if viz_type == 'bar':
            if not colY:
                data_grpd = data.groupby(colX).size().rename('count').reset_index()
                percentage = np.array(data_grpd['count']) / sum(np.array(data_grpd['count']))
                ax = self.axs[self.axs_ind]
                tot_cnt = float(len(data_grpd))
                sns.barplot(x=colX, y="count", data=data_grpd, ax=ax)
            else:
                ax = self.axs[self.axs_ind]
                data[colY] = data[colY].astype('float')
                percentage = np.array(data[colY], dtype=float) / sum(np.array(data[colY], dtype=float))
                sns.barplot(x=colX, y=colY, data=data, ax=ax)
            
            if 'title' in params_keys:
                ax.set_title(params['title'])
            
            for e, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.,
                        height + 3,
                        '%s %s' % (str(round(percentage[e] * 100, 3)), str('%')),
                        ha="center")
        
        if viz_type == 'countplot':
            ''' Prefer when labels are not highly imbalance, The plot would render nicely '''
            ax = self.axs[self.axs_ind]
            tot_cnt = float(len(data))
            sns.countplot(x=colX, hue=label_col, data=data, ax=ax)
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.,
                        height + 5,
                        '%s %s' % (str(round((height / tot_cnt) * 100, 3)), str('%')),
                        ha="center")
        
        if viz_type == 'line':
            ax = self.axs[self.axs_ind]
            #             if 'title' in params:
            #                 assert len(params['title']) == len(data.columns())
            for col_names in data.columns:
                ax.plot(np.array(data[col_names]))
            
            ax.legend(list(data.columns), loc=4)
            
            if 'title' in params_keys:
                ax.set_title(params['title'])
                
                
        if viz_type == 'roc':
            '''
            Send a data frame with two columns
                1) array of false positive values and
                2) array of true positive values
            '''
            ax = self.axs[self.axs_ind]
            ax.plot(np.array(data[colX]), np.array(data[colY]), 'b',
                     label='AUC = %0.2f' % metrics.auc(np.array(data[colX]),np.array(data[colY])))

            ax.legend(loc='lower right')
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
            
            if 'title' in params_keys:
                ax.set_title(params['title'])
            
        self.axs_ind += 1
    
    def show(self):
        plt.show()