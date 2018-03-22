from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
import logging

from CreditCardFraudDetection.utils import Score
from CreditCardFraudDetection.models.nnet import fc_layers
import CreditCardFraudDetection.data_prep as data_prep


logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_loss(which_loss, y, y_pred, lamda = 0, weights_arr=[]):
    '''
    :param which_loss: KL Divergence or Mean Squared error
    :param y: true y
    :param y_pred: predicted y
    :param lamda: the regularization parameter
    :param weights_arr: array of weights to  participate on regularization
    :return:
    
        The input data has a distribution and after the autoencoder module the output data
        would have a distribution. Our objective is to bring the output distribution as close as
        to the input distribution so that the latent neurons or the hidden layers neuron learns a good
        representation.

        KL divergence is good at comparing two distribution, and its differentiable so we use KL divergence here
        as our loss function.
        :return:
    '''
    loss = 0
    loss_ = 0
    with tf.name_scope('regularize'):
        if len(weights_arr) >0:
            print('Regulariztion Activated')
            for weights in weights_arr:
                loss += tf.nn.l2_loss(weights)
        else:
            print ('Regulariztion De-Activated')
    
    with tf.name_scope('loss'):
        if which_loss == 'kl_div':
            loss_ = (rho * tf.log(rho / rho_hat)) + (rho_hat * tf.log((1 - rho) / (1 - rho_hat)))
        elif which_loss == 'mse':
            batch_mse = tf.reduce_mean(tf.pow(y - y_pred, 2), 1) # since axis is 1 the mean is computed per row i.e
            # for each data point in the batch what is the reduced_mean or SSE calculated for all feature
            loss_ = tf.reduce_mean(tf.pow(y - y_pred, 2) + lamda*loss)  # Note no axis is given so this outputs only 1 values
        else:
            raise ValueError('Provide a valid loss function')
        
    return loss_, batch_mse

def autoencoder(layer_dims, learning_rate, lamda, REGULARIZE):
    '''
    :param layer_dims:
    :param learning_rate:
    :param lamda:
    :param REGULARIZE:
    :return:
    The idea of using autoencoder is that we try to learn a representation of the data such that that representation  marks some distinction between the fraud case and the non-fraud case. In-order to learn this representation we calculate the mean squared error between the inp data and the output data (same shape as input).
    
    The output highly depends on the activations selected (tanh and sigmoid). if the output and input are from different distribution ex (input = 200, 300 and output =0.2,0.4) then the weight will be learned such that the linear activation outputs high values. Now when we squash the high linear activation through a tanh unit then the values would be at far end and while backpropagation the gradient wold have a high chance os becoming 0. Here comes vanishing gradients.
    
    So while using a tanh activation, it is a good idea to perform standarization (z-score) of the training data.
    And while using a sigmoid unit one can prefer using a Min-Max scaling.
    ** The dataset should be standarize before head**
    '''
    inpX = tf.placeholder(dtype=tf.float32, shape=[None, 29])
    y_true = inpX
    
    inp_dim = inpX.get_shape().as_list()[-1]
    # print (inp_dim)

    w1, X = fc_layers(inpX, inp=inp_dim, out=layer_dims[0], seed=421, name='hid_1')
    X = tf.nn.tanh(X, name='tanh_1')
    
    # w2, X = fc_layers(X, inp=layer_dims[0], out=layer_dims[1], seed=552, name='hid_2')
    # X = tf.nn.tanh(X, name='relu')
    #
    # w3, X = fc_layers(X, inp=layer_dims[1], out=layer_dims[2], seed=963, name='hid_3')
    # X = tf.nn.tanh(X, name='relu_2')

    w4, X = fc_layers(X, inp=layer_dims[1], out=inp_dim, seed=119, name='out')
    X = tf.nn.tanh(X, name='tanh_4')
    

    with tf.name_scope('loss'):
        if REGULARIZE:
            lossMSE, batchMSE = get_loss(which_loss='mse', y=y_true, y_pred=X,
                                         lamda=0.00001, weights_arr=[w1,w4])
        else:
            lossMSE, batchMSE = get_loss(which_loss='mse', y=y_true, y_pred=X,
                                         lamda=0, weights_arr=[])
        
    with tf.variable_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lossMSE)
    
    return dict(inpX=inpX, loss=lossMSE, batch_mse=batchMSE, optimizer=opt)


def nnet(trainX, trainY, cvalidX, cvalidY, layers, batch_size, display_step, num_epochs, learning_rate, lamda,
         REGULARIZE, MODEL_PATH, MODEL_NAME):
    tr_ls_arr = []
    cv_ls_arr = []
    tr_auc_arr = []
    cv_auc_arr = []
    
    tf.reset_default_graph()
    computation_graph = autoencoder(layers, learning_rate, lamda, REGULARIZE=REGULARIZE)

    save_model = os.path.join(MODEL_PATH, MODEL_NAME)
    saver = tf.train.Saver()
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, num_epochs):
            for batch_num, (batchX, batchY) in enumerate(
                    data_prep.get_batches(X=trainX,Y=trainY,batch_size=batch_size)):
                
                
                # print (np.where(batchY==1)[0], batchY)
                feed_dict = {computation_graph['inpX']: batchX}

                ls, bmse, _ = sess.run([computation_graph['loss'],
                                        computation_graph['batch_mse'],
                                        computation_graph['optimizer']],
                                       feed_dict=feed_dict)
                
            if ((epoch + 1) % display_step) == 0:
                # Get the Training Accuracy for all the Training Data Points
                # Note here we don't evaluate the up-sampled Training set, but the original set
                tr_b_mse, tr_ls = sess.run([computation_graph['batch_mse'], computation_graph['loss']],
                                           feed_dict={computation_graph['inpX']: trainX})
                cv_b_mse, cv_ls = sess.run([computation_graph['batch_mse'], computation_graph['loss']],
                                           feed_dict={computation_graph['inpX']: cvalidX})
                tr_ls_arr.append(tr_ls)
                cv_ls_arr.append(cv_ls)
                
                tr_auc_score = Score.auc(trainY, tr_b_mse)
                cv_auc_score = Score.auc(cvalidY, cv_b_mse)
                logging.info('Epoch = %s, Train Loss = %s, CV Loss = %s, Train AUC = %s, CV AUC = %s',
                             str(epoch+1), str(round(tr_ls, 4)), str(round(cv_ls, 4)),
                             str(round(tr_auc_score, 4)), str(round(cv_auc_score, 4)))

                tr_auc_arr.append(tr_auc_score)
                cv_auc_arr.append(cv_auc_score)
                
        if MODEL_PATH:
             saver.save(sess, save_model)

    return tr_ls_arr, cv_ls_arr, tr_auc_arr, cv_auc_arr
    
    
    

    

    
    
    
    
'''
GAN Notes: The Neural Network we use as a generative model have a number of parameters significantly smaller than the amount of data we train them on


'''
