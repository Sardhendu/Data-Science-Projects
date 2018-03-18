from __future__ import division, print_function, absolute_import
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)
import logging
import numpy as np
import tensorflow as tf
import time

from CreditCardFraudDetection import data_prep
from CreditCardFraudDetection.utils import to_one_hot
from CreditCardFraudDetection.models.boosting import Score
from CreditCardFraudDetection.models.boosting import XGBoost

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

epsilon = 1e-3

classifier = XGBoost()
# seed = [284, 873, 199, 982]



def fc_layers(X, inp, out, seed, name):
    with tf.variable_scope(name):
        w = tf.get_variable(dtype=tf.float32,
                            shape=[inp, out],
                            initializer=tf.truncated_normal_initializer(stddev=0.01, seed=seed),
                            name='w')
        
        b = tf.get_variable(dtype=tf.float32,
                            shape=[out],
                            initializer=tf.constant_initializer(1.0),
                            name='b')
    
    return w, tf.matmul(X, w) + b


def activation(X, name):
    X = tf.nn.tanh(X, name=name)
    return X


def batch_norm(X, axes=[0], scope_name='batch_norm'):
    numOUT = X.get_shape().as_list()[-1]
    with tf.variable_scope(scope_name):
        beta = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(0.0),
                name="b",  # offset (bias)
                trainable=True
        )
        gamma = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="w",  # scale(weight)
                trainable=True)
        
        batchMean, batchVar = tf.nn.moments(X, axes=axes, name="moments")  # the input format is [m,numOUT],
        # and hence we just do the normalization across batches by taking axis as [0]
        
        norm = (X - batchMean) / tf.sqrt(batchVar + epsilon)  # Simply the formula for standarization
        BN = gamma * norm + beta
    return BN


def accuracy(labels, logits, type='training', add_smry=True):
    with tf.name_scope("Accuracy"):
        pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    if add_smry:
        tf.summary.scalar('%s_accuracy' % str(type), acc)
    return acc


def model(layers, learning_rate, lamda, regularize):
    inpX = tf.placeholder(dtype=tf.float32, shape=[None, 29])
    inpY = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    is_training = tf.placeholder(tf.bool)
    
    # print(inpX.get_shape().as_list()[-1], layers[0])
    w1, X = fc_layers(inpX, inp=inpX.get_shape().as_list()[-1], out=layers[0], seed=284, name='layer_1')
    # X = batch_norm(X, axes=[0], scope_name='bn_1')
    X = activation(X, name='tanh_1')
    
    w2, X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[1], seed=873, name='layer_2')
    # X = batch_norm(X, axes=[0], scope_name='bn_2')
    X = activation(X, name='tanh_2')
    
    w3, X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[2], seed=776, name='layer_3')
    # X = batch_norm(X, axes=[0], scope_name='bn_3')
    X = activation(X, name='tanh_3')
    
    w4, logits = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[3], seed=651, name='layer_4')
    probs = tf.nn.softmax(logits, name='softmax')
    
    acc = tf.cond(is_training,
                  lambda: accuracy(labels=inpY, logits=logits, type='training', add_smry=True),
                  lambda: accuracy(labels=inpY, logits=logits, type='validation', add_smry=True)
                  )
    
    with tf.variable_scope("Loss"):
        lossCE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=inpY))
        
        if regularize:
            logging.info('Adding Regularization with lambda (%s) to the Loss Function', str(lamda))
            regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w3)
            lossCE = tf.reduce_mean(lossCE + lamda * regularizers)
    
    with tf.variable_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lossCE)
    
    return dict(inpX=inpX, inpY=inpY, outX=X, is_training=is_training, probabilities=probs, loss=lossCE,
                optimizer=opt,
                accuracy=acc)


def net_boost(layers, batch_size, display_step, num_epochs, learning_rate, lamda, regularize):
    tf.reset_default_graph()
    computation_graph = model(layers, learning_rate, lamda, regularize)
    # print (computation_graph)
    
    dataX, dataY, xFeatures, yLabel = data_prep.feature_transform()
    print('Full data Shape, dataX.shape = %s, dataY.shape = %s, len(xFeatures) = %s, yLabel = %s \n' % (
        str(dataX.shape), str(dataY.shape), str(len(xFeatures)), str(yLabel)))
    
    trainX, trainY, testX, testY, cvalidX, cvalidY = data_prep.data_prep(dataX, dataY)
    print('trainX.shape = %s, trainY.shape = %s, testX.shape = %s, testY.shape = %s, cvalidX.shape = %s, '
          'cvalidY.shape = %s \n'
          % (str(trainX.shape), str(trainY.shape), str(testX.shape), str(testY.shape), str(cvalidX.shape),
             str(cvalidY.shape)))
    
    testY_1hot = to_one_hot(testY)
    trainY_1hot = to_one_hot(trainY)
    cvalidY_1hot = to_one_hot(cvalidY)
    
    print('One-hot conversion shape: testY_1hot.shape = %s, trainY_1hot.shape = %s, cvalidY_1hot=1 = %s \n' % (
        str(testY_1hot.shape), str(trainY_1hot.shape), str(cvalidY_1hot.shape)
    ))
    
    ####  When we have class imbalance problem, then we upscale the minority class using SMOTE
    trX, trY = data_prep.upscale_minority_class(trainX, trainY)
    print('Upscaled Training Data: trX.shape = %s, trY.shape = %s, trY=1 = %s, trY=0 = %s \n' % (
        str(trX.shape), str(trY.shape), str(len(np.where(trY == 1)[0])), str(len(np.where(trY == 0)[0]))
    ))
    
    #######  RUN THE SESSION
    tr_loss_arr = []
    tr_acc_arr = []
    tr_precision_arr = []
    tr_recall_arr = []
    
    cv_loss_arr = []
    cv_acc_arr = []
    cv_precision_arr = []
    cv_recall_arr = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(0, num_epochs):
                
                for batch_num, (batchX, batchY) in enumerate(
                        data_prep.get_batches(X=trX, Y=trY, batch_size=batch_size)):
                    
                    # print('Batch Data: batchX.shape = %s, batchY.shape = %s, batchY=1 = %s, batchY=0 = %s \n' % (
                    #     str(batchX.shape), str(batchY.shape), str(len(np.where(batchY == 1)[0])), str(len(np.where(
                    # batchY == 0)[0]))
                    # ))
                    batchY_ = to_one_hot(batchY)
                    
                    feed_dict = {computation_graph['inpX']: batchX,
                                 computation_graph['inpY']: batchY_,
                                 computation_graph['is_training']: True}
                    
                    _ = sess.run([computation_graph['optimizer']], feed_dict=feed_dict)
                
                if ((epoch + 1) % display_step) == 0:
                    # Get the Training Accuracy for all the Training Data Points
                    # Note here we don't evaluate the up-sampled Training set, but the original set
                    netFeaturesTrain, t_loss = sess.run([computation_graph['outX'],
                                                computation_graph['loss']],
                                            feed_dict={
                                                computation_graph['inpX']: trainX,
                                                computation_graph['inpY']: trainY_1hot,
                                                computation_graph['is_training']: False})
                    
                    
                    # Evaluate ar Cross Validation Data
                    netFeaturesCv, cv_loss = sess.run([computation_graph['outX'],
                                                      computation_graph['loss']],
                                                    feed_dict={
                                                        computation_graph['inpX']: cvalidX,
                                                        computation_graph['inpY']: cvalidY_1hot,
                                                        computation_graph['is_training']: False})
                    
                    

                    start_time = time.time()
                    classifier.fit(netFeaturesTrain, trainY)
                    tot_time = time.time() - start_time
                    print ('Total time taken for fitting data: ', tot_time)
                    
                    trY_pred = classifier.predict(netFeaturesTrain)
                    cvY_pred = classifier.predict(netFeaturesCv)
                    
                    tr_acc = Score.accuracy(trainY, trY_pred)
                    cv_acc = Score.accuracy(cvalidY, cvY_pred)

                    tr_precision = Score.precision(trainY, trY_pred)
                    cv_precision = Score.precision(cvalidY, cvY_pred)

                    tr_recall = Score.recall(trainY, trY_pred)
                    cv_recall = Score.recall(cvalidY, cvY_pred)

    #
                    tr_loss_arr += [t_loss]
                    tr_acc_arr += [tr_acc]
                    tr_precision_arr += [tr_precision]
                    tr_recall_arr += [tr_recall]

                    cv_loss_arr += [cv_loss]
                    cv_acc_arr += [cv_acc]
                    cv_precision_arr += [cv_precision]
                    cv_recall_arr += [cv_recall]

                    logging.info('EPOCH %s ..............................................', str(epoch))

                    logging.info("Training loss = %s, Training acc = %s, Training precision =%s, "
                                 "Training recall = %s, Training auc = %s", str(round(t_loss, 5)),
                                 str(round(tr_acc,5)), str(round(tr_precision, 5)), str(round(tr_recall, 5)),
                                 str(round()))

                    logging.info("Validation loss = %s, Validation acc = %s, Validation precision =%s, "
                                 "Validation recall = %s", str(round(cv_loss, 5)), str(round(cv_acc, 5)),
                                 str(round(cv_precision, 5)), str(round(cv_recall, 5)))
    #
    #         # Evaluate ar Cross Validation Data
    #         tsprobs, ts_loss, tsacc = sess.run([computation_graph['probabilities'],
    #                                             computation_graph['loss'],
    #                                             computation_graph['accuracy']],
    #                                            feed_dict={
    #                                                computation_graph['inpX']: testX,
    #                                                computation_graph['inpY']: testY_1hot,
    #                                                computation_graph['is_training']: False})
    #
    #         tsY_pred = sess.run(tf.argmax(tsprobs, 1))
    #         ts_recall_score = Score.recall(testY, tsY_pred)
    #         ts_precsion_score = Score.precision(testY, tsY_pred)
    #
    #         logging.info('TESTING PERFORMANCE STATISTICS ..................................')
    #         logging.info("Test loss = %s, Test acc = %s, Test precision =%s, "
    #                      "Test recall = %s", str(round(ts_loss, 5)), str(round(tsacc, 5)),
    #                      str(round(ts_precsion_score, 5)), str(round(ts_recall_score, 5)))
    #
    # return (tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr,
    #         cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr,
    #         tsacc, ts_precsion_score, ts_recall_score)





layers = [20, 20, 20, 2]
batch_size = 2048
display_step = 20
num_epochs = 500
learning_rate = 0.005
lamda = 0.1
reg = False

net_boost(layers, batch_size, display_step, num_epochs, learning_rate, lamda, regularize=reg)