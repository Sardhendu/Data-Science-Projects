# Credit Card Fraud

### Overview:

The data set was initially posted in Kaggle, and can be found [HERE](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras/tree/master/data)

The data set contains 284807 records where 284315 are non-fraud transaction and 492. We have a typical class imbalance problem.


### Data Prep:

* Test Data: Comprises of 10% of the total data
* Cross Validation: 10% of data left after removing test data
* Training: The left out data is used for training.
* The Fraud Transactions are upsampled only in the training data using SMOTE (Only for Boosting ).
* For Deep Neural Nets, the data is divided into batches. The size is varied for different runs and 2048 is taken to be a good size. 
* Models are fit on the training data, parameters are adjusted using cross validation score and final performance is obtained using the test data. 

### Models

1. [Random Forest, Gradient Boosting and XGBoost](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/boosting.py): Boosting methods are quite famous since they work on random batches and weak classifiers and are robust to overfitting. For our case, we dont tune the parameters extensively rather boosting methods are used as a bench mark system to evaluate other methods.

    * [Model evaluation](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/model_eval_boost.ipynb)
        * Random Forest: 
            * Training: Accuracy : 1, Recall (Identifying Fraud):
            * CrossValidation: Accuracy : 0.999, Recall (Identifying Fraud): 0.84
            * Test: Accuracy : 0.999, Recall (Identifying Fraud): 0.83
        
        * Gradient Boosting:
            * Training: Accuracy : 0.999, Recall (Identifying Fraud): 1
            * CrossValidation: Accuracy : 0.998, Recall (Identifying Fraud): 0.909
            * Test: Accuracy : 0.9985, Recall (Identifying Fraud): 0.872
            
        * XGBoost: (Could be tuned more extensively)
            * Training: Accuracy : 0.999, Recall (Identifying Fraud): 1
            * CrossValidation: Accuracy : 0.999, Recall (Identifying Fraud): 0.8863
            * Test: Accuracy : 0.9989, Recall (Identifying Fraud): 0.8723
        
        
2. [Deep neural Nets](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/nnet.py): A very simple 4 layers network is implemented, parameters are tuned and performances are compared.
        
    * [Model Evaluation](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/model_eval_deepL.ipynb)
        
        * ANN - 4 layers:
            * Average Training: Accuracy : 0.996, Recall (Identifying Fraud): 0.996
            * Average CrossValidation: Accuracy : 0.999, Recall (Identifying Fraud): 0.8909
            * Average Test: Accuracy : 0.998, Recall (Identifying Fraud): 0.915
            
        <img src="https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/images
        /dl_training.png" width="800" height="300"><img src="https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/images/dl_cross_valid.png" width="800" height="300">
     
3. [Autoencoders](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/models/autoencoders.py): Autoencoders can be thought as an unsupervised learning problem, where the input X is fed to the network. X is transformed through one or many hidden layers and final layer computes the mean squared error (MSE) between the input X and the transformedX. The idea here is the make the **MSE** smaller, by doing so the network would try to learn latent features that would hopefully distinguish **Fraud and Non-Frauds**.

    * [Model Evaluation](https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/model_eval_autoEncoders.ipynb): 
     <img src="https://github.com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/images/AEnc_reconstruction_err_test.png" width="500" height="300"><img src="https://github
     .com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/images/AEnc_test_roc.png" width="250" 
     height="300"><img src="https://github
     .com/Sardhendu/Data-Science-Projects/blob/master/CreditCardFraudDetection/images/AEnc_reconstruction_err_test.png" width="250" 
     height="300">

4. Parallelized Bayesian Networks: (TODO)

REFERENCES:

1. https://uu.diva-portal.org/smash/get/diva2:1150344/FULLTEXT01.pdf
2. https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd 
3. https://github.com/aaxwaz/Fraud-detection-using-deep-learning