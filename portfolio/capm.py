import numpy as np
import tensorflow as tf
import math


class Capm:
    def __init__(self, num_stks):
        self.exp = exp = tf.placeholder(tf.float32, shape = (num_stks), name='exp')
        exp = tf.reshape(exp, shape=(num_stks, 1))
        self.cov = cov = tf.placeholder(tf.float32, shape = (num_stks, num_stks), name='cov')
        self.lambda_coef = lambda_coef = tf.placeholder(tf.float32, shape=(), name='lambda')

        init_w = np.full((num_stks), 1 / num_stks)
        self.w = w = tf.Variable(initial_value=init_w, name='weights', dtype=tf.float32)
        w = tf.reshape(w, shape=(num_stks, 1))
        self.port_exp_pow_2 = port_exp_pow_2 = tf.square(tf.matmul(w, exp, transpose_a=True), name='port_exp_pow_2')
        self.port_var = port_var = tf.matmul(tf.matmul(w, cov, transpose_a=True), w, name='port_var')
        self.sharpe_pow_2 = sharpe_pow_2 = tf.reshape(port_exp_pow_2 / port_var, shape=())

        self.constraint_pow_2 = constraint_pow_2 = tf.square(tf.reduce_sum(tf.abs(w)) - 1, name='constraint')

        self.loss = loss = -sharpe_pow_2 + lambda_coef * constraint_pow_2

        # self.optimizer = optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        self.optimizer = optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        # self.optimizer = optimizer = tf.train.AdamOptimizer()
        # self.gv = gv = optimizer.compute_gradients(loss, var_list=[self.w])
        # self.train = optimizer.apply_gradients(gv)
        self.train = optimizer.minimize(loss)

        self.sess = tf.Session()

    def reset_optimizer(self):
        self.optimizer = optimizer = tf.train.RMSPropOptimizer()
        # self.optimizer = optimizer = tf.train.AdamOptimizer()
        # self.optimizer = optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        self.train = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def init(self):
        print('initializing weights...')
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self, exp, cov, lambda_coef):
        feed_dict = {
            self.exp : exp,
            self.cov : cov,
            self.lambda_coef : lambda_coef
        }
        w, sharpe_pow_2, constraint_pow_2, _ = self.sess.run([self.w, self.sharpe_pow_2, self.constraint_pow_2, self.train], feed_dict)
        try:
            return w, math.sqrt(sharpe_pow_2), math.sqrt(constraint_pow_2)
        except:
            _debug = 0
            pass
