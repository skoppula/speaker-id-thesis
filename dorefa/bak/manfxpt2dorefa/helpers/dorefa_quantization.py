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

def quantize(v, n_bits, name="quantize"):
    scale = tf.pow(2.0, n_bits)
    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
        quantized = tf.round(v * scale) / scale
    clipped = tf.minimum(quantized, scale-1)
    return clipped

def get_quantized_nl(nl, a_bits):
    def quantized_nl(inp, name="quantized_nl"):
        logger.info("Quantizing activation {},{}".format(inp.op.name, a_bits))
        q = quantize(nl(inp), a_bits)
        return tf.identity(q, name='quantized_output')
    return quantized_nl

def get_quantized_w(w_bits):
    def quantize_w(inp, name="quantized_w"):
        logger.info("Quantizing weight {},{}".format(inp.op.name, w_bits))
        x = tf.tanh(inp)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        qv = 2 * quantize(x, w_bits) - 1
        return qv
    return quantize_w
