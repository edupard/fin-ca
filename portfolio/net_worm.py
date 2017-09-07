import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from portfolio.config import get_config


class NetWorm(object):
    def __init__(self):
        self.labels = labels = tf.placeholder(tf.float32, [None], name='labels')
        self.x = x = tf.placeholder(tf.float32, [None, None, 5], name='input')
        self.phase = phase = tf.placeholder(tf.bool, name='phase')

        # keep_prob = tf.placeholder(tf.float32)

        cells = []
        with tf.name_scope('rnn'):
            idx = 0
            for num_units in get_config().LSTM_LAYERS_SIZE:
                scope_name = 'rnn_layer_%d' % idx
                with tf.name_scope(scope_name):
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
                    # cell = tf.contrib.rnn.LSTMCell(num_units)
                    # cell = tf.contrib.rnn.GRUCell(num_units)
                    # cell = tf.contrib.rnn.DropoutWrapper(
                    #     cell, output_keep_prob=keep_prob)
                    cells.append(cell)
                idx += 1
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            # Batch size x time steps x features.
            output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

            last_idx = tf.shape(output)[1] - 1
            output = tf.transpose(output, [1, 0, 2])
            self.last = last = tf.gather(output, last_idx, name='rnn_features')

        x = last
        idx = 0
        for num_units in get_config().FC_LAYERS_SIZE:
            scope_name = 'fc_layer_%d' % idx
            with tf.name_scope(scope_name):
                self.h1 = h1 = tf.contrib.layers.fully_connected(x, num_units, activation_fn=None,
                                                                 scope='dense_%d' % idx)
                if get_config().BATCH_NORM:
                    self.h2 = h2 = tf.contrib.layers.batch_norm(h1,
                                                                center=True, scale=True,
                                                                is_training=phase,
                                                                scope='bn_%d' % idx)
                else:
                    h2 = h1
                self.h3 = x = tf.nn.tanh(h2, 'tanh_%d' % idx)
            idx += 1

        # final fc layer
        scope_name = 'fc_layer_%d' % idx
        with tf.name_scope(scope_name):
            self.h = x = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.nn.tanh, scope='dense_%d' % idx)

        with tf.name_scope('loss'):
            # x = tf.transpose(x, [1, 0])
            self.z = z = tf.stop_gradient(
                tf.maximum(tf.reduce_sum(tf.abs(x)), tf.constant(get_config().MIN_PARTITION_Z)),
                name='partition')
            self.weights = weights = tf.reshape(x / z, [-1])
            self.pl = tf.reduce_sum(weights * labels)
            self.cost = -self.pl
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer()
                self.optimizer = optimizer.minimize(self.cost)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    def init(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def fit(self, x, labels):
        self.sess.run((self.optimizer), feed_dict={self.x: x, self.labels: labels, self.phase: True})

    def eval(self, x, labels):
        pl, weights, last, h1, h2, h3, h, z = self.sess.run(
            (self.pl, self.weights, self.last, self.h1, self.h2, self.h3, self.h, self.z),
            feed_dict={self.x: x, self.labels: labels, self.phase: False})
        return pl, weights

    def save_weights(self, path, epoch):
        self.saver.save(self.sess, path, global_step=epoch)

    def load_weights(self, path, epoch):
        self.saver.restore(self.sess, "%s-%d" % (path, epoch))
