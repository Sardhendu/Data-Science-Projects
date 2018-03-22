from __future__ import division, print_function, absolute_import
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)
import logging
import numpy as np
import tensorflow as tf

from CreditCardFraudDetection import data_prep
from CreditCardFraudDetection.utils import to_one_hot, Score



logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w", format="%(asctime)-15s %(levelname)-8s %(message)s")


epsilon = 1e-3

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
        tf.summary.scalar('%s_accuracy'%str(type), acc)
    return acc
    



class DeepNet():
    def __init__(self, learning_rate, lamda, learning_rate_decay, REGULARIZE, LEARNING_RATE_DECAY, BATCH_NORM):
        
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.lr_decay = learning_rate_decay
        self.REGULARIZE = REGULARIZE
        self.LEARNING_RATE_DECAY = LEARNING_RATE_DECAY
        self.BATCH_NORM = BATCH_NORM
        
        

    def model(self, layers, batch_size, train_size):
        inpX = tf.placeholder(dtype=tf.float32, shape=[None, 29])
        inpY = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        is_training = tf.placeholder(tf.bool)
        
        # print(inpX.get_shape().as_list()[-1], layers[0])
        w1, X = fc_layers(inpX, inp=inpX.get_shape().as_list()[-1], out=layers[0], seed=284, name='layer_1')
        if self.BATCH_NORM:
            X = batch_norm(X, axes=[0], scope_name='bn_1')
        X = activation(X, name='tanh_1')
        
        w2, X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[1], seed=873, name='layer_2')
        if self.BATCH_NORM:
            X = batch_norm(X, axes=[0], scope_name='bn_2')
        X = activation(X, name='tanh_2')
        
        w3, X = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[2], seed=776, name='layer_3')
        if self.BATCH_NORM:
            X = batch_norm(X, axes=[0], scope_name='bn_3')
        X = activation(X, name='tanh_3')
        
        w4, logits = fc_layers(X, inp=X.get_shape().as_list()[-1], out=layers[3], seed=651, name='layer_4')
        probs = tf.nn.softmax(logits, name='softmax')
    
        acc = tf.cond(is_training,
                           lambda: accuracy(labels=inpY, logits=logits, type='training', add_smry=True),
                           lambda: accuracy(labels=inpY, logits=logits, type='validation', add_smry=True)
                           )
        
        with tf.variable_scope("Loss"):
            lossCE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=inpY))
    
            if self.REGULARIZE:
                logging.info('Adding Regularization with lambda (%s) to the Loss Function', str(self.lamda))
                regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
                lossCE = tf.reduce_mean(lossCE + self.lamda * regularizers)
            
        with tf.variable_scope("optimizer"):
            if self.LEARNING_RATE_DECAY:
                globalStep = tf.Variable(0, trainable=False)
                l_rate = tf.train.exponential_decay(self.learning_rate,
                                                    globalStep * batch_size,  # Used for decay computation
                                                    train_size,  # Decay steps
                                                    self.lr_decay,  # Decay rate
                                                    staircase=True)  # Will decay the learning rate in discrete interval
                opt = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(lossCE, global_step=globalStep)
            else:
                l_rate = self.learning_rate
                opt = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(lossCE)
        
            
            
        return dict(inpX=inpX, inpY=inpY, is_training=is_training, probabilities=probs, loss=lossCE, optimizer=opt,
                    accuracy=acc, learning_rate=l_rate)
    
    
    
       
        
    def nnet(self, layers, batch_size, num_epochs, display_step):
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
        
        train_size = trX.shape[0]
        tf.reset_default_graph()
        computation_graph = self.model(layers, batch_size, train_size)
        
        #######  RUN THE SESSION
        tr_loss_arr = []
        tr_acc_arr = []
        tr_precision_arr = []
        tr_recall_arr = []
        tr_auc_arr =[]
    
        cv_loss_arr = []
        cv_acc_arr = []
        cv_precision_arr = []
        cv_recall_arr = []
        cv_auc_arr = []
        
        l_rate_arr = []
    
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
    
                for epoch in range(0,num_epochs):
    
                    for batch_num, (batchX, batchY) in enumerate(data_prep.get_batches(X=trX, Y=trY,
                                                                                       batch_size=batch_size)):
                    # print('Batch Data: batchX.shape = %s, batchY.shape = %s, batchY=1 = %s, batchY=0 = %s \n' % (
                    #     str(batchX.shape), str(batchY.shape), str(len(np.where(batchY == 1)[0])), str(len(np.where(batchY == 0)[0]))
                    # ))
                        batchY_ = to_one_hot(batchY)
    
                        feed_dict = {computation_graph['inpX'] : batchX,
                                     computation_graph['inpY'] : batchY_,
                                     computation_graph['is_training'] :True}
    
                        _, lrate = sess.run([computation_graph['optimizer'], computation_graph['learning_rate']],
                                             feed_dict=feed_dict)

                        l_rate_arr += [lrate]
    
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
                        t_auc_score = Score.auc(trainY, y_pred)
    
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
                        v_auc_score = Score.auc(cvalidY, cvY_pred)
    
                        tr_loss_arr += [t_loss]
                        tr_acc_arr += [tacc]
                        tr_precision_arr += [t_precsion_score]
                        tr_recall_arr += [t_recall_score]
                        tr_auc_arr += [t_auc_score]
    
                        cv_loss_arr += [v_loss]
                        cv_acc_arr += [vacc]
                        cv_precision_arr += [v_precsion_score]
                        cv_recall_arr += [v_recall_score]
                        cv_auc_arr += [v_auc_score]
                        
                        logging.info('EPOCH %s ..............................................' ,str(epoch))
    
                        logging.info("Training loss = %s, Training acc = %s, Training precision =%s, "
                                     "Training recall = %s, Training auc = %s", str(round(t_loss, 5)),
                                     str(round(tacc, 5)), str(round(t_precsion_score, 5)), str(round(t_recall_score, 5)),
                                     str(round(t_auc_score)))
                        
                        logging.info("Validation loss = %s, Validation acc = %s, Validation precision =%s, "
                                     "Validation recall = %s, Validataion auc = %s",str(round(v_loss, 5)), str(round(vacc,5)),
                                     str(round(v_precsion_score, 5)), str(round(v_recall_score, 5)), str(round(
                                    v_auc_score)))
    
    
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
                ts_auc_score = Score.auc(testY, tsY_pred)
                
                logging.info('TESTING PERFORMANCE STATISTICS ..................................')
                logging.info("Test loss = %s, Test acc = %s, Test precision =%s, "
                             "Test recall = %s", str(round(ts_loss, 5)), str(round(tsacc, 5)),
                             str(round(ts_precsion_score, 5)), str(round(ts_recall_score, 5)))
                
        return (l_rate_arr, tr_loss_arr, tr_acc_arr, tr_precision_arr, tr_recall_arr, tr_auc_arr,
                cv_loss_arr, cv_acc_arr, cv_precision_arr, cv_recall_arr, cv_auc_arr,
                tsacc, ts_precsion_score, ts_recall_score, ts_auc_score)
    
    
