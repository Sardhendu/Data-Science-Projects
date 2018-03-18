from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import numpy  as np
import pandas as pd
import time

from CreditCardFraudDetection.plots import Plot
import xgboost as xgb


from CreditCardFraudDetection import data_prep
from xgboost.sklearn import XGBClassifier





class RandomForest():
    def __init__(self, model_obj=None):
        if model_obj:
            self.clf_rf = model_obj
        else:
            self.clf_rf = RandomForestClassifier(n_estimators=1500, random_state=12)
        
    def fit(self, X, Y):
        self.clf_rf.fit(X, Y)
    
    def predict(self, X):
        return self.clf_rf.predict(X)


class GradientBoosting():
    def __init__(self, model_obj=None):
        '''
        Tree Based.
            1. max_feature: Number of feature to be consider while deciding each split
            2. min_sample_split: example if 70 - A node can only split if it has at least 70 samples
                    — mostly (take 0.5-1% of the total), but take lower value if class imbalance problem else higher is fine
            3. min_sample_leaf/min_weight_fraction_leaf: example if 30 - A node would not allow its leaf node to have less than 30 samples.
            4. max_depth: Split only till a depth
            5. max_leaf_nodes: maximum number of leaf nodes in a tree.
                - max_leaf_nodes and max_depth can be used interchangeable since only binary trees are created depth (
                n) = 2^n (leaves). If both 4 and 5 are defined then 5 is preferred over 4
            6. max_features: The number of features to consider while searching for a best split. As a rule of thumb sq_root(total_features) would be a good selection. 
        
        Boosting based:
            1. learning_rate: never very large.  — 0.2 - 0.05
            2. n_estimators: How many trees: Tuned using CV  — 40-70
            3. subsample: Percentage of data used as a random sample. Generally 0.8 it preferred, but this is again tunable using CV

        '''
        if model_obj:
            self.clf_gb = model_obj
        else:
            self.clf_gb = GradientBoostingClassifier(
                    learning_rate=0.005,
                    n_estimators=1500,              # how many trees start with 30 and go till 1000
                    max_depth=9,                    # 5-20 often a good choice
                    min_samples_split=1200,
                    min_samples_leaf=60,
                    subsample=0.85,
                    random_state=10,
                    max_features=7,
                    warm_start=True)
        
    def fit(self, X, Y):
        self.clf_gb.fit(X, Y)
    
    def predict(self, X):
        return self.clf_gb.predict(X)
        
 
class XGBoost():
    def __init__(self, model_obj=None):
        
        '''
        Credits:
        https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        
        Benifits over normal boosting
        1. Regularization: xgboost has a regularization affect that normal boosting (GB) doesnt have
        2. Parallel Processing: Since the output of 1 tree is required to identify weak learners in the next
        tree. The parallelism could be based on Features and node.
        3. Flexibility/ handling missing values
        4. Unlike GB classifiers, it fits the tree completely and then prunes.
        
        General Parameters:
        1. gbtree -> for tree based model, gblinear -> for linear based model
        2. nthreads -> default to max number of threads available
        
        
        Boosted Parameters:
        1. eta : learning_rate, I guess it uses similar to weight decay technique by shrinking learning_rate every
        iteration.
        2. min_child_weight: min sum of weights for all observation required in a child for the node to split.
        Basically every feature has a weight w and if the child feature weight adds to a min value that means we can
        split. Higher values of min_child_weight can be treated as regularization (remember adding W^2 in L2 norm).
        But very high values would lead to model underfit.
        3. max_depth: maximum depth of a tree
        4. max_leaf_node: maximum number of terminal nodes in the tree
        5. gamma: In tree based ML methods a node is split only when the split decreases the loss function. Gamma
        specifies the minimum loss reduction for split. The value can vary based on loss function and hence should be tuned.
        6. max_delta_step: Read further (useful during class imbalance)
        7. subsample: Fraction of observation to be randomly sampled
        8. col_sample_bytree: Fraction of columns to be randomly sampled by each tree {0.5-1}
        9. lambda: L2 regularization on weights
        10. alpha: L1 regularization on weights
        
        Evaluation Parameters:
        1. rmse: Root mean squared error
        2. mae: mean absolute error
        3. logloss: negative loglikelihood
        4. error: binary classification error rate
        5. merror: Multiclass classification error rate
        6. mlogloss: multiclass logloss
        7. auc : area under curve
        
        '''
        
        if model_obj:
            self.clf_xgb = model_obj
        else:
            self.clf_xgb = XGBClassifier(
                            learning_rate=0.1,
                            n_estimators=10,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='binary:logistic',
                            nthread=4,
                            scale_pos_weight=1,
                            seed=27)
        
    def fit(self, X, Y):
        self.clf_xgb.fit(X, Y)

    def predict(self, X):
        return self.clf_xgb.predict(X)
    
    def predict_probs(self, X):
        return self.clf_xgb.predict_proba(X)[:,1]
    
    def feature_importance(self, xFeatures, is_plot):
        idx_desc = np.argsort(self.clf_xgb.feature_importances_)[::-1]
        xFeatures = np.array(xFeatures, dtype=str)[idx_desc]
        feature_coeff = np.array(self.clf_xgb.feature_importances_, dtype=float)[idx_desc]
        dataOUT = pd.DataFrame(np.column_stack((xFeatures, feature_coeff)),columns=['features','feature_coeff'])
        if is_plot:
            pl = Plot()
            pl.vizualize(data=dataOUT, colX='features', colY='feature_coeff', viz_type='bar')
            pl.show()
        return dataOUT

def main():
    dataX, dataY, xFeatures, yLabel = data_prep.feature_transform()
    trainX, trainY, testX, testY, cvalidX, cvalidY = data_prep.data_prep(dataX, dataY)
    print (len(np.where(cvalidY==0)[0]), len(np.where(cvalidY==1)[0]))

    trX, trY = data_prep.upscale_minority_class(trainX=trainX, trainY=trainY)
    
    
    # classifier = RandomForest()
    # classifier = GradientBoosting()
    classifier = XGBoost()
    
    ## Random Forest
    
    start_time = time.time()
    classifier.fit(trX, trY)
    tot_time = time.time() - start_time
    
    trY_pred = classifier.predict(trX)
    cvY_pred = classifier.predict(cvalidX)
    tsY_pred = classifier.predict(testX)
    dataOUT = classifier.feature_importance(xFeatures, is_plot=True)

    print('Time taken to fit: ', str(tot_time))
    print('Train - Recall Score: ', Score.recall(trY, trY_pred))
    print('Train - Recall Score reversed: ', Score.recall(trY, trY_pred, reverse=True))
    print('Cvalid - Recall Score: ',Score.recall(cvalidY, cvY_pred))
    print('Cvalid - Recall Score reversed: ',Score.recall(cvalidY, cvY_pred, reverse=True))
    print('Test - Recall Score: ', Score.recall(testY, tsY_pred))
    print('Test - Recall Score reversed: ',Score.recall(testY, tsY_pred, reverse=True))

    # Random Forest
    # Time taken to fit:  5194.590213060379
    # Train - Recall Score:  1.0
    # Train - Recall Score reversed:  0.0
    # Cvalid - Recall Score:  0.840909090909
    # Cvalid - Recall Score reversed:  0.000234475751299
    # Test - Recall Score:  0.829787234043
    # Test - Recall Score reversed:  0.000316522473096
    
    # Gradient Boosting Method: took appx 1.30 hrs to complete, But worth the try
    # Train - Recall Score: 1.0
    # Train - Recall Score reversed: 0.000794643322391
    # Cvalid - Recall Score:: 0.909090909091
    # Cvalid - Recall Score reversed: 0.00121145804838
    # Test - Recall Score: 0.872340425532
    # Test - Recall Score reversed: 0.00112541323767

    # XGBoost
    # Time taken to fit:  36.861494064331055
    # Train - Recall Score:  0.93499557084
    # Train - Recall Score reversed:  0.99038177618
    # Cvalid - Recall Score:  0.909090909091
    # Cvalid - Recall Score reversed:  0.989917542694
    # Test - Recall Score:  0.936170212766
    # Test - Recall Score reversed:  0.989906450025

# main()