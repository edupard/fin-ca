import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from portfolio.config import get_config
import numpy as np


class NetTurtle(object):
    def __init__(self):
        print('creating neural network...')
        self.labels = labels = tf.placeholder(tf.float32, [None, None, 1], name='labels')
        self.input = input = tf.placeholder(tf.float32, [None, None, 5], name='input')

        cells = []
        with tf.name_scope('rnn'):
            idx = 0
            for num_units in get_config().LSTM_LAYERS_SIZE:
                scope_name = 'rnn_layer_%d' % idx
                with tf.name_scope(scope_name):
                    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
                    cells.append(rnn_cell)
                idx += 1
            self.rnn_cell = rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)

            state = ()
            for s in rnn_cell.state_size:
                c = tf.placeholder(tf.float32, [None, s.c])
                h = tf.placeholder(tf.float32, [None, s.h])
                state += (tf.contrib.rnn.LSTMStateTuple(c, h),)
            self.state = state

            # Batch size x time steps x features.
            output, new_state = tf.nn.dynamic_rnn(rnn_cell, input, initial_state=state)
            self.new_state = new_state

        # final fc layer
        with tf.name_scope('fc_layer'):
            self.returns = tf.contrib.layers.fully_connected(output, 1, activation_fn=None, scope='dense_%d' % idx)

        with tf.name_scope('loss'):
            diff = self.returns - labels
            self.cost = tf.reduce_mean(tf.square(diff))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    def zero_state(self, batch_size):
        zero_state = ()
        for s in self.rnn_cell.state_size:
            c = np.zeros((batch_size, s.c))
            h = np.zeros((batch_size, s.h))
            zero_state += (tf.contrib.rnn.LSTMStateTuple(c, h),)
        return zero_state

    def init(self):
        print('initializing weights...')
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)



    def fill_feed_dict(self, feed_dict, state):
        idx = 0
        for s in state:
            feed_dict[self.state[idx].c] = s.c
            feed_dict[self.state[idx].h] = s.h
            idx += 1

    def eval(self, state, input, labels):
        feed_dict = {self.input: input, self.labels: labels}
        self.fill_feed_dict(feed_dict, state)

        new_state, cost, returns = self.sess.run((self.new_state, self.cost, self.returns), feed_dict)
        return new_state, cost, returns

    def fit(self, state, input, labels):
        feed_dict = {self.input: input, self.labels: labels}
        self.fill_feed_dict(feed_dict, state)
        new_state, cost, returns, _ = self.sess.run((self.new_state, self.cost, self.returns, self.optimizer), feed_dict)
        return new_state, cost, returns

    def save_weights(self, path, epoch):
        print('saving weights after %d epoch' % epoch)
        self.saver.save(self.sess, path, global_step=epoch)

    def load_weights(self, path, epoch):
        print('loading %d epoch weights' % epoch)
        self.saver.restore(self.sess, "%s-%d" % (path, epoch))
