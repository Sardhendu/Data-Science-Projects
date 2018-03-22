# Credit Card Fraud

### Overview:

The data set was initially posted in Kaggle, and can be found [HERE](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras/tree/master/data)

The data set contains 284807 records where 284315 are non-fraud transaction and 492. We have a typical class imbalance problem.


### Data Prep:

* Test Data: Comprises of 10% of the total data
* Cross Validation: 10% of data left after removing test data
* Training: The left out data is used for training.
* The Fraud Transactions are upsampled only in the training data using SMOTE.
* For Deep Neural Nets, the data is divided into batches. The size is varied for different runs and 2048 is taken to be a good size. 
* Models are fit on the training data, parameters are adjusted using cross validation score and final performance is obtained using the test data. 

### Models

1. [Random Forest, Gradient Boosting and XGBoost](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/boosting.py): Boosting methods are quite famous since they work on random batches and weak classifiers. The parameters are not tuned extensively rather boosting methods are used as a bench mark.

    * [Model evaluation](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/model_eval_boost.ipynb)

2. [Deep neural Nets](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/nnet.py): A very simple 4 layers network is implemented, parameters are tuned and performances are compared.

    * [Model Evaluation on a differnt set of parameters](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/model_eval_deepL.ipynb)
    
     <img src="https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/images/Screen%20Shot%202018-03-18%20at%202.27.49%20AM.png" width="1000" height="300">
     
3. [Autoencoders](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/autoencoders.py):



REFERENCES:

1. https://uu.diva-portal.org/smash/get/diva2:1150344/FULLTEXT01.pdf
2. https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd 
3. https://github.com/aaxwaz/Fraud-detection-using-deep-learning