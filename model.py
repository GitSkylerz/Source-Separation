import tensorflow as tf
import numpy as np


def DTLoss(targets, logits, n_src):
    loss = 0
    for i in range(n_src):
        for j in range(n_src):
            if i == j:
                loss += tf.losses.mean_squared_error(targets[..., i], logits[..., j])
            else:
                loss += 1 / \
                (tf.losses.mean_squared_error(targets[..., i], logits[..., j]) + \
                 np.finfo(np.float32).min)
    return loss


# def less(x):
#     cond = tf.less(x, tf.zeros(tf.shape(x)))
#     y    = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    
#     return y


class MaskCRNN(object):
    
    def __init__(self, t_bins, f_bins, n_mic, n_src):
        # parameters
        self.t_bins     = t_bins      # number of time axis bin
        self.f_bins     = f_bins      # number of freq axis bin
        self.n_mic      = n_mic       # number of channel
        self.n_src      = n_src       # number of target
        # self.decay_step = decay_step  # global step to decay learning rate

        # build net & initialize variable
        self._graph = tf.Graph()
        self.__build()
        
        
    def __build(self):
        with self._graph.as_default():
            with tf.variable_scope("inputs"):
                # input arguments
                self._features = tf.placeholder(tf.float32, shape=[None, self.t_bins, self.f_bins, 2*self.n_mic])
                self._targets  = tf.placeholder(tf.float32, shape=[None, self.t_bins, self.f_bins, 2*self.n_src])
                global_step    = tf.Variable(1, trainable=False)
                
                # preprocessing
                feature_magnitude = self._features[..., :self.n_mic]
                target_magnitude  = self._targets[..., :self.n_src]
                feature_phase     = self._features[..., self.n_mic:]
                target_phase      = self._targets[..., self.n_src:]
                phase_difference  = target_phase - self._features[..., self.n_mic:self.n_mic+1]
                target_real       = target_magnitude * tf.cos(phase_difference)
                target_image      = target_magnitude * tf.sin(phase_difference)
            
            with tf.variable_scope("cnn", initializer=tf.keras.initializers.Orthogonal(gain=1.0),
                                          regularizer=tf.contrib.layers.l2_regularizer(scale=1e-6)):
                conv = tf.layers.conv2d(feature_magnitude, 64, (3, 3), (1, 1), 
                                        "same", activation=tf.nn.relu)
                conv = tf.layers.max_pooling2d(conv, [1, 4], [1, 4], "valid")
                conv = tf.layers.conv2d(conv, 64, (3, 3), (1, 1), 
                                        "same", activation=tf.nn.relu)
                conv = tf.layers.max_pooling2d(conv, [1, 2], [1, 2], "valid")
                conv = tf.layers.conv2d(conv, 64, (3, 3), (1, 1), 
                                        "same", activation=tf.nn.relu)
                conv = tf.layers.max_pooling2d(conv, [1, 2], [1, 2], "valid")
                conv = tf.reshape(conv, (-1, self.t_bins, 64*32))
                conv = tf.unstack(conv, axis=1)
            
            with tf.variable_scope("rnn", initializer=tf.variance_scaling_initializer(), 
                                          regularizer=tf.contrib.layers.l2_regularizer(scale=1e-6)):
                # cells formation
                cells_forward  = []
                cells_backward = []
                for i in range(3):
                    cell = tf.nn.rnn_cell.GRUCell(num_units=1024)
                    cells_forward.append(cell)
                    cell = tf.nn.rnn_cell.GRUCell(num_units=1024)
                    cells_backward.append(cell)

                # rnn formation
                rnn_forward  = tf.nn.rnn_cell.MultiRNNCell(cells=cells_forward)
                rnn_backward = tf.nn.rnn_cell.MultiRNNCell(cells=cells_backward)
                rnn, _, _ = tf.nn.static_bidirectional_rnn(rnn_forward, rnn_backward, conv, dtype=tf.float32)
                rnn = tf.stack(rnn, axis=1)
                rnn = tf.reshape(rnn, [-1, self.t_bins, 2048])
            
            with tf.variable_scope("fnn", initializer=tf.variance_scaling_initializer(), 
                                          regularizer=tf.contrib.layers.l2_regularizer(scale=1e-6)):
                fnn = tf.layers.dense(rnn, units=self.f_bins*self.n_src)
                fnn = tf.nn.relu(fnn)
            
            with tf.variable_scope("mask", initializer=tf.keras.initializers.Orthogonal(gain=1.0),
                                           regularizer=tf.contrib.layers.l2_regularizer(1e-6)):
                # mask for real part
                mask_real = tf.layers.dense(fnn, units=self.f_bins*self.n_src)
                mask_real = tf.reshape(mask_real, [-1, self.t_bins, self.f_bins, self.n_src])
                # mask_rv = 1 - tf.reduce_sum(mask_re, axis=-1, keepdims=True)
                # self._mask_re = tf.concat([mask_re, mask_rv], axis=-1)

                # mask for imag part
                mask_image = tf.layers.dense(fnn, units=self.f_bins*self.n_src)
                mask_image = tf.reshape(mask_image, [-1, self.t_bins, self.f_bins, self.n_src])
                # mask_iv = 1 - tf.reduce_sum(mask_im, axis=-1, keepdims=True)
                # self._mask_im = tf.concat([mask_im, mask_iv], axis=-1)

            with tf.variable_scope("outputs"):
                # logits layer
                logits_real  = mask_real  * feature_magnitude[..., :1]
                logits_image = mask_image * feature_magnitude[..., :1]
                self._logits = tf.concat((logits_real, logits_image), axis=-1)
                # logit_re     = tf.concat((logits_real, mask_rv * ftr_mgt[..., :1]), axis=-1)
                # logit_im     = tf.concat((logit_im, mask_iv * ftr_mgt[..., :1]), axis=-1)
                
                # regression: MSE & L2-regularization & permutational loss
                self._loss = DTLoss(target_real, logits_real, self.n_src) + \
                             DTLoss(target_image, logits_image, self.n_src) + \
                             tf.losses.get_regularization_loss()
                
                # backward
                '''
                lr = tf.train.exponential_decay(
                    learning_rate=1e-3,
                    global_step=global_step,
                    decay_steps=self.decay_step,
                    decay_rate=0.1,
                    staircase=True
                )
                '''
                lr = 1e-3
                optimizer     = tf.train.AdamOptimizer(lr)
                self.minimize = optimizer.minimize(loss=self._loss)

                # operation
                self._session = tf.Session()
                self._session.run(tf.global_variables_initializer())
                
                
    def _fit(self, x, y):
        
        feed_dict={self._features: x,
                   self._targets:  y}
        self._session.run(self.minimize, feed_dict=feed_dict)  
        
        
    def _eval(self, x, y):
        
        feed_dict={self._features: x,
                   self._targets:  y}
        
        return self._session.run([self._logits, self._loss], feed_dict=feed_dict)
    
    def _pred(self, x):
        
        feed_dict={self._features: x}
        
        return self._session.run([self._logits, self._mask_real, self._mask_image], 
                                 feed_dict=feed_dict)
        
        
    def _save(self, path):
        
        with self._graph.as_default():
            
            self.saver = tf.train.Saver()
            
        self.saver.save(self._session, path) # save_path ex. './Parameter/myVGG_Clean/myVGG_Clean.ckpt'
        
        print("Successfully save the model !")

        
    def _restore(self, path, keyword=False):
        
        with self._graph.as_default():
            
            if keyword:
                var_list = [i for i in tf.global_variables() if keyword in i.name]
                self.saver = tf.train.Saver(var_list=var_list)
            else:
                self.saver = tf.train.Saver()
                
        self.saver.restore(self._session, tf.train.latest_checkpoint(path)) # load_path ex. './Parameter/myVGG_Clean/'
