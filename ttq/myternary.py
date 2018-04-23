#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

G = tf.get_default_graph()

def p_ternarize(x, p):

    x = tf.tanh(x)
    shape = x.get_shape()

    thre = tf.get_variable('T', trainable=False, collections=[tf.GraphKeys.VARIABLES, 'thresholds'],
            initializer=0.05)
    flat_x = tf.reshape(x, [-1])
    k = int(flat_x.get_shape().dims[0].value * (1 - p))
    topK, _ = tf.nn.top_k(tf.abs(flat_x), k)
    update_thre = thre.assign(topK[-1])
    tf.add_to_collection('update_thre_op', update_thre)

    mask = tf.zeros(shape)
    mask = tf.where((x > thre) | (x < -thre), tf.ones(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask)

    tf.histogram_summary(w.name, w)
    return w

def get_tw(thre):
    def tw_ternarize(x):

        shape = x.get_shape()

        thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

        w_p = tf.get_variable('Wp', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)
        w_n = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'negatives'], initializer=1.0)

        tf.summary.scalar(w_p.name, w_p)
        tf.summary.scalar(w_n.name, w_n)

        mask = tf.ones(shape)
        mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
        mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
        mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)


        num_p = tf.cast(tf.count_nonzero(tf.where(x > thre_x, mask, tf.zeros(shape))), tf.float32)
        num_n = tf.cast(tf.count_nonzero(tf.where(x < -thre_x, mask, tf.zeros(shape))), tf.float32)
        num_zeros = tf.cast(tf.count_nonzero(tf.where((x < thre_x) & (x > - thre_x), mask, tf.zeros(shape))), tf.float32)
        num_elements = tf.cast(tf.size(x, out_type=tf.int64), tf.float32)

        tf.summary.scalar(x.name + '_sparsity', num_zeros/num_elements)
        tf.summary.scalar(x.name + '_percent_p', num_p/num_elements)
        tf.summary.scalar(x.name + '_percent_n', num_n/num_elements)

        with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
            w =  tf.sign(x) * tf.stop_gradient(mask_z)

        w = w * mask_np

        tf.summary.histogram(w.name, w)
        return w

    return tw_ternarize
