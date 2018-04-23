#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.utils.develop import log_deprecated
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils import symbolic_functions as symbf
from tensorflow.contrib.framework import add_model_variable
from tensorpack.tfutils import get_current_tower_context
from custom_layers_with_mult_tracking import *
from tensorflow.python.training import moving_averages

import tensorflow as tf
import math

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def compute_nbits_for_maxval_tf(inp, overflow_rate):
    abst = tf.reshape(tf.abs(inp), [-1])
    split_idx = tf.floor(tf.maximum(1.0, tf.constant(overflow_rate, dtype=tf.float32) * tf.cast(tf.size(inp), tf.float32)))
    values, indices = tf.nn.top_k(abst, k=tf.cast(split_idx, tf.int32))
    return tf.ceil(log2(values[-1] + 1e-12))

def linear_quantize_tf(v, n_bits, bits_per_delta_tensor, name="quantized_nl"):
    bound = math.pow(2.0, n_bits-1)
    min_val, max_val = - bound, bound - 1
    delta = tf.pow(2.0, bits_per_delta_tensor)
    
    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
        quantized = tf.round(v / delta)
    clipped = tf.clip_by_value(quantized, min_val, max_val)
    return tf.identity(clipped*delta, name='quantized_output')

def quantize_var_tf(v, n_bits, overflow_rate=0.01, qtype='linear'):
    # +1 bc we get extra bit by taking abs val and seperating pos/neg
    bits_per_maxval = compute_nbits_for_maxval_tf(v, overflow_rate)
    bits_per_maxval = tf.identity(bits_per_maxval, name='bits_for_maxval')
    bits_per_maxval_var = tf.get_variable('bits_for_maxval_var', [], trainable=False)
    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, bits_per_maxval_var)
    
    # if get_current_tower_context().is_training or get_current_tower_context().is_main_training_tower:
    assign_op = moving_averages.assign_moving_average(bits_per_maxval_var, bits_per_maxval, name='assign_bits_per_maxval_ema', decay=0.1, zero_debias=False)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_op)

    if qtype == 'linear':
        bits_per_delta = bits_per_maxval + 1. - n_bits
        return linear_quantize_tf(v, n_bits, bits_per_delta)

def get_quantized_nl(nl, n_bits, overflow_rate=0.01, qtype='linear'):
    def quantized_nl(inp, name="quantized_nl"):
        return quantize_var_tf(nl(inp), n_bits, overflow_rate, qtype)
    return quantized_nl
