from __future__ import division, print_function, absolute_import

import warnings
from models import Score
warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)

import numpy as np
import tensorflow as tf
import data_prep


layer_shape = [128,128,128,2]
seed = [284, 873, 199, 982]



def fc_layers(X, inp, out, seed, name):
    with tf.variable_scope(name):
        w = tf.get_variable(dtype=tf.float32,
                            shape=[inp, out],
                            initializer=tf.truncated_normal_initializer(stddev=0.1, seed = seed),
                            name='w')
        
        b = tf.get_variable(dtype=tf.float32,
                            shape=[out],
                            initializer=tf.constant_initializer(1.0),
                            name='b')

    return tf.matmul(X, w) + b
   
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
    
    
def to_one_hot(y):
    y = np.array(y, dtype=int)
    n_values = int(np.max(y)) + 1
    y = np.eye(n_values)[y]
    return y


def model():
    inpX = tf.placeholder(dtype=tf.float32, shape=[None, 29])
    inpY = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    is_training = tf.placeholder(tf.bool)
    
    
    
    # print(inpX.get_shape().as_list()[-1], layer_shape[0])
    X = fc_layers(inpX, inp=inpX.get_shape().as_list()[-1], out=layer_shape[0], seed=284, name='layer_1')
    X = activation(X, name='tanh_1')
    X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layer_shape[1], seed=873, name='layer_2')
    X = activation(X, name='tanh_2')
    X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layer_shape[2], seed=776, name='layer_3')
    X = activation(X, name='tanh_3')
    
    logits = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layer_shape[3], seed=651, name='layer_4')
    probs = tf.nn.softmax(logits, name='softmax')

    acc = tf.cond(is_training,
                       lambda: accuracy(labels=inpY, logits=logits, type='training', add_smry=True),
                       lambda: accuracy(labels=inpY, logits=logits, type='validation', add_smry=True)
                       )
    
    with tf.variable_scope("Loss"):
        lossCE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=inpY))
        
    with tf.variable_scope("optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(lossCE)
            
    return dict(inpX=inpX, inpY=inpY, is_training=is_training, probabilities=probs, loss=lossCE, optimizer=opt,
                accuracy=acc)



   
    
def nnet():
    training_graph = model()
    # print (training_graph)

    dataX, dataY, xFeatures, yLabel = data_prep.feature_transform()
    trainX, trainY, testX, testY, cvalidX, cvalidY = data_prep.data_prep(dataX, dataY)
    
    trX, trY = data_prep.upscale_minority_class(trainX, trainY)
    print('Training Shape: ', trX.shape, trY.shape)
    print('Validation Shape: ', cvalidX.shape, cvalidY.shape)
    print('Test Shape: ', testX.shape, testY.shape)
    
    testY = to_one_hot(testY)
    cvalidY_1hot = to_one_hot(cvalidY)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(0,100):
                
                for num_batches, (batchX, batchY) in enumerate(data_prep.get_batches(trX, trY)):
                    # print (batchY)
                    # print (batchX.shape, batchY.shape)
                    batchY_ = to_one_hot(batchY)
                    # print(batchX.shape, batchY.shape)
                    # print (batchX[0:5])
                    # print ('')
                    # print (np.column_stack((batchY[0:5], batchY[len(batchY)-5:len(batchY)])))
                    
                    feed_dict = {training_graph['inpX'] : batchX,
                                 training_graph['inpY'] : batchY_,
                                 training_graph['is_training'] :True}
                    
                    probs, losses, _, tacc = sess.run([training_graph['probabilities'],
                                                     training_graph['loss'],
                                                     training_graph['optimizer'],
                                                     training_graph['accuracy']],
                                                     feed_dict=feed_dict)

                    
                    
                    if ((num_batches+1) % 1000) == 0:
                        y_pred = sess.run(tf.argmax(probs, 1))
                        recall_score = Score.recall(batchY, y_pred)
                        precsion_score = Score.precision(batchY, y_pred)
                        print('Training Accuracy: ', tacc)
                        print('Training Precision: ', precsion_score)
                        print('Training Recall: ', recall_score)
                        print ('')
                        
                    
                feed_dict = {training_graph['inpX']: cvalidX,
                             training_graph['inpY']: cvalidY_1hot,
                             training_graph['is_training']: False}

                vprobs, vacc = sess.run([training_graph['probabilities'],
                                       training_graph['accuracy']],
                                       feed_dict=feed_dict)

                cvY_pred = sess.run(tf.argmax(vprobs, 1))
                # print(len(cvY_pred), cvY_pred, len(np.where(cvY_pred == 0)[0]), len(np.where(cvY_pred == 1)[0]))
                recall_score = Score.recall(cvalidY, cvY_pred)
                precsion_score = Score.precision(cvalidY, cvY_pred)
                print('CrossValidation Accuracy: ', vacc)
                print('CrossValidation Precision: ', precsion_score)
                print('CrossValidation Recall: ', recall_score)
                print ('')
                # #     break
                # break
            
    
nnet()


