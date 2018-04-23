#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized


@graph_memoized
def get_dorefa(bitW, bitA):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x):
        if bitW == 32:
            return x
        if bitW == 1:   # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    return fw, fa
