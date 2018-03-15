from __future__ import division, print_function, absolute_import
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)
import logging
import numpy as np
import tensorflow as tf

import data_prep
from utils import to_one_hot
from models import Score


logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



'''
To Do's :
L2 Regularization
Batch Normalization
Learning Rate Decay

Start the experiment with 1 hidden Layer and few nodes and then increase the
number of hidden layer and layer nodes


Baseline by https://uu.diva-portal.org/smash/get/diva2:1150344/FULLTEXT01.pdf
4 Hidden layer with 20 nodes each
Converges at epoch 200 and learning rate 0.005
Batch size = 2048
'''



# seed = [284, 873, 199, 982]



def fc_layers(X, inp, out, seed, name):
    with tf.variable_scope(name):
        w = tf.get_variable(dtype=tf.float32,
                            shape=[inp, out],
                            initializer=tf.truncated_normal_initializer(stddev=0.01, seed = seed),
                            name='w')
        
        b = tf.get_variable(dtype=tf.float32,
                            shape=[out],
                            initializer=tf.constant_initializer(1.0),
                            name='b')

    return w, tf.matmul(X, w) + b
   
def activation(X, name):
    X = tf.nn.tanh(X, name=name)
    return X

def accuracy(labels, logits, type='training', add_smry=True):
    with tf.name_scope("Accuracy"):
        pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    if add_smry:
        tf.summary.scalar('%s_accuracy'%str(type), acc)
    return acc
    

def model(layers, learning_rate, lamda, regularize):
    inpX = tf.placeholder(dtype=tf.float32, shape=[None, 29])
    inpY = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    is_training = tf.placeholder(tf.bool)
    
    # print(inpX.get_shape().as_list()[-1], layers[0])
    w1, X = fc_layers(inpX, inp=inpX.get_shape().as_list()[-1], out=layers[0], seed=284, name='layer_1')
    X = activation(X, name='tanh_1')
    w2, X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[1], seed=873, name='layer_2')
    X = activation(X, name='tanh_2')
    w3, X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[2], seed=776, name='layer_3')
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
            
    return dict(inpX=inpX, inpY=inpY, is_training=is_training, probabilities=probs, loss=lossCE, optimizer=opt,
                accuracy=acc)



   
    
def nnet(layers, batch_size, display_step, num_epochs, learning_rate, lamda, regularize):
    tf.reset_default_graph()
    computation_graph = model(layers, learning_rate, lamda, regularize)
    # print (computation_graph)

    dataX, dataY, xFeatures, yLabel = data_prep.feature_transform()
    print('Full data Shape, dataX.shape = %s, dataY.shape = %s, len(xFeatures) = %s, yLabel = %s \n'%(
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
    print ('Upscaled Training Data: trX.shape = %s, trY.shape = %s, trY=1 = %s, trY=0 = %s \n' %(
        str(trX.shape), str(trY.shape), str(len(np.where(trY ==1)[0])), str(len(np.where(trY==0)[0]))
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

            for epoch in range(0,num_epochs):

                for batch_num, (batchX, batchY) in enumerate(data_prep.get_batches(X=trX, Y=trY, batch_size=batch_size)):
                # print('Batch Data: batchX.shape = %s, batchY.shape = %s, batchY=1 = %s, batchY=0 = %s \n' % (
                #     str(batchX.shape), str(batchY.shape), str(len(np.where(batchY == 1)[0])), str(len(np.where(batchY == 0)[0]))
                # ))
                    batchY_ = to_one_hot(batchY)

                    feed_dict = {computation_graph['inpX'] : batchX,
                                 computation_graph['inpY'] : batchY_,
                                 computation_graph['is_training'] :True}

                    _ = sess.run([computation_graph['optimizer']], feed_dict=feed_dict)

                if ((epoch+1) % display_step) == 0:

                    # Get the Training Accuracy for all the Training Data Points
                    # Note here we don't evaluate the up-sampled Training set, but the original set
                    tprobs, t_loss, tacc = sess.run([computation_graph['probabilities'],
                                                   computation_graph['loss'],
                                                   computation_graph['accuracy']],
                                                   feed_dict={
                                                       computation_graph['inpX']: trainX,
                                                       computation_graph['inpY']: trainY_1hot,
                                                       computation_graph['is_training']: False})




                    y_pred = sess.run(tf.argmax(tprobs, 1))
                    t_recall_score = Score.recall(trainY, y_pred)
                    t_precsion_score = Score.precision(trainY, y_pred)

                    # Evaluate ar Cross Validation Data
                    vprobs, v_loss, vacc = sess.run([computation_graph['probabilities'],
                                                   computation_graph['loss'],
                                                   computation_graph['accuracy']],
                                                   feed_dict={
                                                       computation_graph['inpX']: cvalidX,
                                                       computation_graph['inpY']: cvalidY_1hot,
                                                       computation_graph['is_training']: False})

                    cvY_pred = sess.run(tf.argmax(vprobs, 1))
                    v_recall_score = Score.recall(cvalidY, cvY_pred)
                    v_precsion_score = Score.precision(cvalidY, cvY_pred)

                    tr_loss_arr += [t_loss]
                    tr_acc_arr += [tacc]
                    tr_precision_arr += [t_precsion_score]
                    tr_recall_arr += [t_recall_score]

                    cv_loss_arr += [v_loss]
                    cv_acc_arr += [vacc]
                    cv_precision_arr += [v_precsion_score]
                    cv_recall_arr += [v_recall_score]
                    
                    logging.info('EPOCH %s ..............................................' ,str(epoch))

                    logging.info("Training loss = %s, Training acc = %s, Training precision =%s, "
                                 "Training recall = %s", str(round(t_loss, 5)), str(round(tacc, 5)),
                                 str(round(t_precsion_score, 5)), str(round(t_recall_score, 5)))
                    
                    logging.info("Validation loss = %s, Validation acc = %s, Validation precision =%s, "
                                 "Validation recall = %s", str(round(v_loss, 5)), str(round(vacc, 5)),
                                 str(round(v_precsion_score, 5)), str(round(v_recall_score, 5)))


            # Evaluate ar Cross Validation Data
            tsprobs, ts_loss, tsacc = sess.run([computation_graph['probabilities'],
                                             computation_graph['loss'],
                                             computation_graph['accuracy']],
                                            feed_dict={
                                                computation_graph['inpX']: testX,
                                                computation_graph['inpY']: testY_1hot,
                                                computation_graph['is_training']: False})

            tsY_pred = sess.run(tf.argmax(tsprobs, 1))
            ts_recall_score = Score.recall(testY, tsY_pred)
            ts_precsion_score = Score.precision(testY, tsY_pred)
            
            logging.info('TESTING PERFORMANCE STATISTICS ..................................')
            logging.info("Test loss = %s, Test acc = %s, Test precision =%s, "
                         "Test recall = %s", str(round(ts_loss, 5)), str(round(tsacc, 5)),
                         str(round(ts_precsion_score, 5)), str(round(ts_recall_score, 5)))
            
    return (tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr,
            cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr,
            tsacc, ts_precsion_score, ts_recall_score)


