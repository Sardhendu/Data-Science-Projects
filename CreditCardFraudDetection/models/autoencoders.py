
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from CreditCardFraudDetection.models.nnet import fc_layers
import CreditCardFraudDetection.data_prep as data_prep

def get_loss(which_loss, y, y_pred):
    '''
        The input data has a distribution and after the autoencoder module the output data
        would have a distribution. Our objective is to bring the output distribution as close as
        to the input distribution so that the latent neurons or the hidden layers neuron learns a good
        representation.

        KL divergence is good at comparing two distribution, and its differentiable so we use KL divergence here
        as our loss function.
        :return:
    '''
    with tf.name_scope('loss'):
        if which_loss == 'kl_div':
            l = (rho * tf.log(rho / rho_hat)) + (rho_hat * tf.log((1 - rho) / (1 - rho_hat)))
        elif which_loss == 'mse':
            l = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
        else:
            raise ValueError('Provide a valid loss function')
    return l


def autoencoder(layer_dims, learning_rate, lamda, regularize):
    inpX = tf.placeholder(dtype=tf.float32, shape=[None, 29])

    inp_dim = inpX.get_shape().as_list()[-1]
    print (inp_dim)

    w1, X = fc_layers(inpX, inp=inp_dim, out=layer_dims[0], seed=421, name='hid_1')
    X = tf.nn.tanh(X, name='tanh_1')
    
    w2, X = fc_layers(X, inp=layer_dims[0], out=layer_dims[1], seed=552, name='hid_2')
    X = tf.nn.relu(X, name='relu_1')

    w3, X = fc_layers(X, inp=layer_dims[1], out=layer_dims[2], seed=963, name='hid_3')
    X = tf.nn.tanh(X, name='tanh_2')

    w4, X = fc_layers(X, inp=layer_dims[2], out=inp_dim, seed=119, name='out')
    X = tf.nn.relu(X, name='relu_2')
    
    with tf.name_scope('loss'):
        lossMSE = get_loss(which_loss='mse', y=inpX, y_pred=X)
        
    with tf.variable_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lossMSE)
    
    
    return dict(inpX=inpX, loss=lossMSE, optimizer=opt)


def nnet(layers, batch_size, display_step, num_epochs, learning_rate, lamda, regularize):
    tf.reset_default_graph()
    computation_graph = autoencoder(layers, learning_rate, lamda, regularize)
    # print (computation_graph)
    
    dataX, dataY, xFeatures, yLabel = data_prep.feature_transform()
    print('Full data Shape, dataX.shape = %s, dataY.shape = %s, len(xFeatures) = %s, yLabel = %s \n' % (
        str(dataX.shape), str(dataY.shape), str(len(xFeatures)), str(yLabel)))
    
    trainX, trainY, testX, testY, cvalidX, cvalidY = data_prep.data_prep(dataX, dataY)
    print('trainX.shape = %s, trainY.shape = %s, testX.shape = %s, testY.shape = %s, cvalidX.shape = %s, '
          'cvalidY.shape = %s \n'
          % (str(trainX.shape), str(trainY.shape), str(testX.shape), str(testY.shape), str(cvalidX.shape),
             str(cvalidY.shape)))
    
    # Combine the test and validation data set
    testX_nw = np.vstack((testX, cvalidX))
    testY_nw = np.append(testY, cvalidY)
    print ('testX_nw.shape = %s, testY_nw.shape = %s '%(str(testX_nw.shape), str(testY_nw.shape)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, num_epochs):

            for batch_num, (batchX, batchY) in enumerate(
                    data_prep.get_batches(X=trainX,Y=trainY,batch_size=batch_size)):
                # print (np.where(batchY==1)[0], np.where(batchY==0)[0])
                feed_dict = {computation_graph['inpX']: batchX}

                ls, _ = sess.run([computation_graph['loss'], computation_graph['optimizer']], feed_dict=feed_dict)

            if ((epoch + 1) % display_step) == 0:
                # Get the Training Accuracy for all the Training Data Points
                # Note here we don't evaluate the up-sampled Training set, but the original set
                t_loss = sess.run(computation_graph['loss'],
                                    feed_dict={
                                        computation_graph['inpX']: trainX})
                
                print(t_loss)

    #                 y_pred = sess.run(tf.argmax(tprobs, 1))
    #                 t_recall_score = Score.recall(trainY, y_pred)
    #                 t_precsion_score = Score.precision(trainY, y_pred)
    #
    #                 # Evaluate ar Cross Validation Data
    #                 vprobs, v_loss, vacc = sess.run([computation_graph['probabilities'],
    #                                                  computation_graph['loss'],
    #                                                  computation_graph['accuracy']],
    #                                                 feed_dict={
    #                                                     computation_graph['inpX']: cvalidX,
    #                                                     computation_graph['inpY']: cvalidY_1hot,
    #                                                     computation_graph['is_training']: False})
    #
    #                 cvY_pred = sess.run(tf.argmax(vprobs, 1))
    #                 v_recall_score = Score.recall(cvalidY, cvY_pred)
    #                 v_precsion_score = Score.precision(cvalidY, cvY_pred)
    #
    #                 tr_loss_arr += [t_loss]
    #                 tr_acc_arr += [tacc]
    #                 tr_precision_arr += [t_precsion_score]
    #                 tr_recall_arr += [t_recall_score]
    #
    #                 cv_loss_arr += [v_loss]
    #                 cv_acc_arr += [vacc]
    #                 cv_precision_arr += [v_precsion_score]
    #                 cv_recall_arr += [v_recall_score]
    #
    #                 logging.info('EPOCH %s ..............................................', str(epoch))
    #
    #                 logging.info("Training loss = %s, Training acc = %s, Training precision =%s, "
    #                              "Training recall = %s", str(round(t_loss, 5)), str(round(tacc, 5)),
    #                              str(round(t_precsion_score, 5)), str(round(t_recall_score, 5)))
    #
    #                 logging.info("Validation loss = %s, Validation acc = %s, Validation precision =%s, "
    #                              "Validation recall = %s", str(round(v_loss, 5)), str(round(vacc, 5)),
    #                              str(round(v_precsion_score, 5)), str(round(v_recall_score, 5)))
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



    
debugg = True
if debugg:
    nnet(layers=[14, 7, 7], batch_size=32, display_step=1, num_epochs=100, learning_rate=0.1, lamda=0.01,
         regularize=True)
    
    
    
    
'''
GAN Notes: The Neural Network we use as a generative model have a number of parameters significantly smaller than the amount of data we train them on


'''
