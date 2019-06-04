# -*- coding: utf-8 -*-

import tensorflow as tf


class PredictModel(object):
    def __init__(self, holder_dim, win_size, hidden_dim, batch_size, epoch_num):
        self.holder_dim = holder_dim
        self.win_size = win_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        self.build()

    def build(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name="input x")
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name="input y")
        with tf.variable_scope("Encoder Section"):
            W = tf.get_variable(name="W",
                                shape=[self.holder_dim, self.hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.hidden_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

        with tf.variable_scope("Decoder Section"):
            W = tf.get_variable(name="W",
                                shape=[self.hidden_dim, self.holder_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.holder_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

    def train(self, train, dev):
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        pass


class EncoderModel(object):
    def __init__(self):
        self.build()

    def build(self):
        pass


if __name__ == '__main__':
    pass
