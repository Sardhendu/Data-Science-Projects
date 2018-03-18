# Credit Card Fraud

### Overview:

The data set was initially posted in Kaggle, and can be found [HERE](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras/tree/master/data)

The data set contains 284807 records where 284315 are non-fraud transaction and 492. We have a typical class imbalance problem.

### Models

1. [Random Forest, Gradient Boosting and XGBoost](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/boosting.py): Boosting methods are quite famous since they work on random batches and weak classifiers. The parameters were not tuned extensiely rather boosting methods are used as a bench mark.

2. [Deep neural Nets](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/nnet.py): A very simple 4 layers network is implemented, parameters are tuned and performances are compared.

   