from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import unison_shuffled_copies

# records_per_label_per_batch =
# batch_size_per_class = 512
smote_batch_size = 50



def upscale_minority_class(trainX, trainY):
    print('Input trainX.shape=%s and trainY.shape=%s' % (str(trainX.shape), str(trainY.shape)))
    sm = SMOTE(random_state=376, ratio=1.0)
    trainX, trainY = sm.fit_sample(trainX, trainY)
    trainX, trainY = unison_shuffled_copies(trainX, trainY)
    return trainX, trainY



def get_batches_upsample_every_batch(trainX, trainY, batch_size_per_class):
    np.random.seed(183)
    np.random.shuffle(trainY)
    
    class_0_indices = np.where(trainY == 0)[0]
    class_1_indices = np.where(trainY == 1)[0]
    
    class_0_len = len(class_0_indices)
    class_1_len = len(class_1_indices)
    
    # FOR LARGER CLASS
    class_0_num_batches = int(np.ceil(class_0_len / batch_size_per_class))
    class_0_extra_smaple = np.random.choice(class_0_indices, class_0_num_batches * batch_size_per_class - class_0_len,
                                            replace=False)
    class_0_indices = np.append(class_0_indices, class_0_extra_smaple)
    # print('Larger Class: Num of batches: ', class_0_num_batches)
    # print('Extra Samples class: ', len(class_0_extra_smaple))
    # print('New Upsampled Data Size: ', len(class_0_indices))
    
    
    # FOR SMALLER CLASS
    smote_augmentation_len = smote_batch_size * class_0_num_batches
    upsampling_class_1_len = len(class_0_indices) - smote_augmentation_len
    class_1_indices = np.random.choice(class_1_indices, upsampling_class_1_len, replace=True)
    if (len(class_1_indices) + smote_augmentation_len) != len(class_0_indices):
        raise ValueError('Mismatch is data length')
    
    sm = SMOTE(random_state=12, ratio=1.0)
    # BATCH GENERATION:
    for batch_num in range(0, class_0_num_batches):
        class_0_from = batch_num * batch_size_per_class
        class_0_to = (batch_num * batch_size_per_class) + batch_size_per_class
        class_1_from = batch_num * (batch_size_per_class - smote_batch_size)
        class_1_to = (batch_num * (batch_size_per_class - smote_batch_size)) + (batch_size_per_class - smote_batch_size)
        # print(class_0_from, class_0_to, class_1_from, class_1_to)
        
        batch_0_X = trainX[class_0_indices[class_0_from:class_0_to]]
        batch_0_Y = trainY[class_0_indices[class_0_from:class_0_to]]
        
        batch_1_X = trainX[class_1_indices[class_1_from:class_1_to]]
        batch_1_Y = trainY[class_1_indices[class_1_from:class_1_to]]
        
        batchX = np.vstack((batch_0_X, batch_1_X))
        batchY = np.append(batch_0_Y, batch_1_Y)
        batchX, batchY = sm.fit_sample(batchX, batchY)
        # print (batch_0_X.shape, batch_1_X.shape, batch_0_Y.shape, batch_1_Y.shape, batchX.shape, batchY.shape)
        #     print (len(np.where(batchY_ == 0)[0]), len(np.where(batchY_ == 1)[0]))
        #     if batch_num ==20:
        yield batchX, batchY


def get_batches_augment(trainX, trainY, batch_size_per_class):
    # Shuffle the training batches
    np.random.seed(183)
    np.random.shuffle(trainY)
    
    class_0_indices = np.where(trainY == 0)[0]
    class_1_indices = np.where(trainY == 1)[0]
    # print (len(class_0_indices), len(class_1_indices))
    
    class_0_len = len(class_0_indices)
    class_1_len = len(class_1_indices)
    
    if class_0_len != class_1_len:
        raise ValueError('Unbalanced class problem %s Vs %s. Make sure to level them with SMOTE' % (str(class_0_len),
                                                                                                    str(class_1_len)))
    
    class_0_num_batches = int(np.ceil(class_0_len / batch_size_per_class))
    class_0_extra_smaple = np.random.choice(class_0_indices, class_0_num_batches * batch_size_per_class - class_0_len,
                                            replace=False)
    class_1_extra_smaple = np.random.choice(class_1_indices, class_0_num_batches * batch_size_per_class - class_0_len,
                                            replace=False)
    
    class_0_indices = np.append(class_0_indices, class_0_extra_smaple)
    class_1_indices = np.append(class_1_indices, class_1_extra_smaple)
    # print('Class 0: len indices: ', len(class_0_indices))
    # print('Class 1: len indices: ', len(class_1_indices))
    
    
    # BATCH GENERATION:
    for batch_num in range(0, class_0_num_batches):
        from_idx = batch_num * batch_size_per_class
        to_idx = (batch_num * batch_size_per_class) + batch_size_per_class
        
        batch_0_X = trainX[class_0_indices[from_idx:to_idx]]
        batch_0_Y = trainY[class_0_indices[from_idx:to_idx]]
        
        batch_1_X = trainX[class_1_indices[from_idx:to_idx]]
        batch_1_Y = trainY[class_1_indices[from_idx:to_idx]]
        
        batchX = np.vstack((batch_0_X, batch_1_X))
        batchY = np.append(batch_0_Y, batch_1_Y)
        
        yield batchX, batchY
        
        
def get_batches(X, Y, batch_size):
    Y = Y.reshape(-1,1)

    num_batches = int(np.ceil(len(X) / batch_size))
    for batch_num in range(0, num_batches):
        if batch_num != (num_batches - 1):
            from_idx = batch_num * batch_size
            to_idx = (batch_num * batch_size) + batch_size
        else:
            from_idx = batch_num * batch_size
            to_idx = (batch_num * batch_size) + (len(X)-(batch_num * batch_size))

        yield X[from_idx : to_idx], Y[from_idx : to_idx].flatten()

def data_prep(dataX, dataY):
    # Get Test Data
    trainX, testX, trainY, testY = train_test_split(
            dataX, dataY, test_size=0.1, random_state=873)

    # Get Cross-Validation Data
    trainX, cvalidX, trainY, cvalidY = train_test_split(
            trainX, trainY, test_size=0.1, random_state=231)

    trainX = np.array(trainX, dtype='float32')
    trainY = np.array(trainY, dtype='int32')
    testX = np.array(testX, dtype='float32')
    testY = np.array(testY, dtype='int32')
    cvalidX = np.array(cvalidX, dtype='float32')
    cvalidY = np.array(cvalidY, dtype='int32')
    return trainX, trainY, testX, testY, cvalidX, cvalidY


def feature_transform():
    data_dir = '/Users/sam/All-Program/App-DataSet/z-others/creditcard.csv'
    data = pd.read_csv(data_dir)
    data = data.drop(['Time'], axis=1)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    dataX = data.drop(['Class'], axis=1)
    dataY = data[['Class']]
    
    xFeatures = dataX.columns
    yLabel = dataY.columns
    
    return dataX, dataY, xFeatures, yLabel


debugg = False
if debugg:
    trainX, trainY, testX, testY, cvalidX, cvalidY = data_prep()
    trainX, trainY = upscale_minority_class(trainX, trainY)
    for batchX, batchY in get_batches(trainX, trainY):
        print (batchX.shape, batchY.shape)
        print (np.column_stack((batchY[0:5], batchY[len(batchY)-5: len(batchY)])))
        break