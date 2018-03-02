#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from custom_layers_with_mult_tracking import *
from tensorpack.models.nonlin import *
from tensorpack.tfutils.symbolic_functions import *

def fcn1_net(inp, bn=True, context=50):
    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44681.pdf
    # sized so that total number of weights is 400000 is 230
    with argscope(FullyConnectedWithTrackedMults, out_dim=256, nl=(BNReLU if bn else tf.nn.relu)):
        l = FullyConnectedWithTrackedMults('linear0', inp)
        l = FullyConnectedWithTrackedMults('linear1', l)
        l = FullyConnectedWithTrackedMults('linear2', l)
        l = FullyConnectedWithTrackedMults('linear3', l)
    return l

def fcn2_net(inp, bn=True, context=50):
    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44681.pdf
    # sized so that total number of weights is 400000 is 230
    with argscope(FullyConnectedWithTrackedMults, out_dim=504, nl=(BNReLU if bn else tf.nn.relu)):
        l = FullyConnectedWithTrackedMults('linear0', inp)
        l = FullyConnectedWithTrackedMults('linear1', l)
        l = FullyConnectedWithTrackedMults('linear2', l)
        l = FullyConnectedWithTrackedMults('linear3', l)
    return l

def cnn_net(inp, bn=True, context=50):
    # https://pdfs.semanticscholar.org/ef8d/6c4c65a9a227f63f857fcb789db4202f2180.pdf
    l = tf.reshape(inp, (-1, context, 20, 1))
    l = Conv2DWithTrackedMults('conv0', l, 6, kernel_shape=5, stride=5, nl=(BNReLU if bn else tf.nn.relu))
    with argscope(FullyConnectedWithTrackedMults, out_dim=256, nl=(BNReLU if bn else tf.nn.relu)):
        l = FullyConnectedWithTrackedMults('linear0', l)
        l = FullyConnectedWithTrackedMults('linear1', l)
        l = FullyConnectedWithTrackedMults('linear2', l)
    return l

def bnorm(x, name=None):
    return BatchNorm('bn', x)

def maxout1_net(inp, bn=True, context=50):
    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf
    l = FullyConnectedWithTrackedMults('linear0',inp,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 4)
    l = Dropout(l, keep_prob=0.4)
    l = FullyConnectedWithTrackedMults('linear1',l,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 4)
    l = Dropout(l, keep_prob=0.4)
    l = FullyConnectedWithTrackedMults('linear2',l,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 4)
    l = FullyConnectedWithTrackedMults('linear3',l,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 4)
    return l

def maxout2_net(inp, bn=True, context=50):
    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf
    l = FullyConnectedWithTrackedMults('linear0',inp,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 2)
    l = Dropout(l, keep_prob=0.4)

    l = FullyConnectedWithTrackedMults('linear1',l,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 2)
    l = Dropout(l, keep_prob=0.4)

    l = FullyConnectedWithTrackedMults('linear2',l,nl=(bnorm if bn else tf.identity), out_dim=1000)
    l = Maxout(l, 2)
    return l


def lcn_net(inp, bn=True, context=50):
    # https://pdfs.semanticscholar.org/ef8d/6c4c65a9a227f63f857fcb789db4202f2180.pdf
    l = tf.reshape(inp, (-1, context, 20, 1))
    with argscope(Conv2DWithTrackedMults, out_channel=25, kernel_shape=10, nl=(BNReLU if bn else tf.nn.relu), padding='valid'):
        # batch flatten within FCWithTrackedMults
        l0a = Conv2DWithTrackedMults('conv0a', l[:,0:10,0:10,:])
        l0b = Conv2DWithTrackedMults('conv0b', l[:,0:10,10:20,:])
        l0c = Conv2DWithTrackedMults('conv0c', l[:,10:20,0:10,:])
        l0d = Conv2DWithTrackedMults('conv0d', l[:,10:20,10:20,:])
        l0e = Conv2DWithTrackedMults('conv0e', l[:,20:30,0:10,:])
        l0f = Conv2DWithTrackedMults('conv0f', l[:,20:30,10:20,:])
        l0g = Conv2DWithTrackedMults('conv0g', l[:,30:40,0:10,:])
        l0h = Conv2DWithTrackedMults('conv0h', l[:,30:40,10:20,:])
        l0i = Conv2DWithTrackedMults('conv0i', l[:,40:50,0:10,:])
        l0j = Conv2DWithTrackedMults('conv0j', l[:,40:50,10:20,:])
    l0 = tf.concat([l0a, l0b, l0c, l0d, l0e, l0f, l0g, l0h, l0i, l0j], 1)
    with argscope(FullyConnectedWithTrackedMults, out_dim=256, nl=(BNReLU if bn else tf.nn.relu)):
        l1 = FullyConnectedWithTrackedMults('linear1', l0)
        l2 = FullyConnectedWithTrackedMults('linear2', l1)
        l3 = FullyConnectedWithTrackedMults('linear3', l2)
    return l3

def dsc1_net(inp, bn=True, context=50):
    l = tf.reshape(inp, (-1, context, 20, 1))
    l = DepthwiseSeparableConvWithTrackedMults('conv0', l, 16, nl=BNReLU)
    with argscope(FullyConnectedWithTrackedMults, out_dim=256, nl=(BNReLU if bn else tf.nn.relu)):
        l = FullyConnectedWithTrackedMults('linear0', l)
        l = FullyConnectedWithTrackedMults('linear1', l)
        l = FullyConnectedWithTrackedMults('linear2', l)
    return l

def dsc2_net(inp, bn=True, context=50):
    l = tf.reshape(inp, (-1, context, 20, 1))
    with argscope(DepthwiseSeparableConvWithTrackedMults, nl=(BNReLU if bn else tf.nn.relu)):
        l = DepthwiseSeparableConvWithTrackedMults('conv0', l, 4)
        l = DepthwiseSeparableConvWithTrackedMults('conv1', l, 16, downsample=True)
        l = DepthwiseSeparableConvWithTrackedMults('conv2', l, 32, downsample=True)
    with argscope(FullyConnectedWithTrackedMults, out_dim=256, nl=(BNReLU if bn else tf.nn.relu)):
        l = FullyConnectedWithTrackedMults('linear0', l, out_dim=256, nl=BNReLU)
        l = FullyConnectedWithTrackedMults('linear1', l, out_dim=256, nl=BNReLU)
        l = FullyConnectedWithTrackedMults('linear2', l, out_dim=256, nl=BNReLU)
    return l
