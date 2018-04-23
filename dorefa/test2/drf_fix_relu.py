#cti!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: svhn-digit-dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.dataflow import dataset
from tensorpack.tfutils.varreplace import remap_variables
import tensorflow as tf

from dorefa import get_dorefa
import numpy as np
import multiprocessing
import argparse
import socket
import os

from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.utils.develop import log_deprecated


from helpers.custom_layers_with_mult_tracking import *
from helpers.rsr2015 import *
from helpers.helpers import get_tensors_from_graph
from helpers.helpers import DumpTensorsOnce
from helpers.helpers import str2bool
from helpers.baselinearchs import *
from tensorpack.tfutils.varreplace import remap_variables

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

BITW = 32
BITA = 32

class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()
        
        self.n_spks = 255
        self.n_context = 50
        self.net_fn = fcn2_net
        self.network_complexity = {'mults':0, 'weights':0}
        self.regularize = True
        
        
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, self.n_context*20], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        inp, label = inputs
        is_training = get_current_tower_context().is_training

        fw, fa = get_dorefa(BITW, BITA)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            if not (name.endswith('W') or name.endswith('b')) or 'linear0' in name or 'last_linear' in name:
                print("Not quantizing", name)
                return v
            else:
                logger.info("Quantizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x, name="activate"):
            return fa(tf.nn.relu(BNWithTrackedMults(x)))

        with remap_variables(binarize_weight), \
                argscope([FullyConnectedWithTrackedMults], network_complexity=self.network_complexity), \
                argscope([BNReLUWithTrackedMults], network_complexity=self.network_complexity), \
                argscope([BNWithTrackedMults], network_complexity=self.network_complexity), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4):
            l = self.net_fn(inp, nonlin, self.n_context)
            logits = FullyConnectedWithTrackedMults('last_linear', l, out_dim=self.n_spks, nl=tf.identity)

        prob = tf.nn.softmax(logits, name='output')

        # used for validation accuracy of utterance
        identity_guesses = flatten(tf.argmax(prob, axis=1))
        uniq_identities, _, count = tf.unique_with_counts(identity_guesses)
        idx_to_identity_with_most_votes = tf.argmax(count)
        chosen_identity = tf.gather(uniq_identities, idx_to_identity_with_most_votes)
        wrong = tf.expand_dims(tf.not_equal(chosen_identity, tf.cast(label[0], tf.int64)), axis=0, name='utt-wrong')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        add_moving_summary(cost)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        with tf.name_scope('original-weight-summaries'):
            add_param_summary(('.*/W', ['rms', 'histogram']))
            add_param_summary(('.*/b', ['rms', 'histogram']))

        with tf.name_scope('activation-summaries'):
            def fn(name):
                return (name.endswith('output') or name.endswith('output:0')) and "Inference" not in name and 'quantized' not in name
            tensors = get_tensors_from_graph(tf.get_default_graph(), fn) 
            print("Adding activation tensors to summary:", tensors)
            for tensor in tensors:
                add_tensor_summary(tensor, ['rms', 'histogram'])

        if self.regularize:
            # decreasing regularization on all W of fc layers
            wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(), 480000, 0.2, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            add_moving_summary(wd_cost)
            self.cost = tf.add_n([cost, wd_cost], name='cost')
        else:
            self.cost = tf.identity(cost, name='cost')

        tf.constant([self.network_complexity['mults']], name='TotalMults')
        tf.constant([self.network_complexity['weights']], name='TotalWeights')
        logger.info("Parameter count: {}".format(self.network_complexity))

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=4721 * 100,
            decay_rate=0.5, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def create_dataflow(partition, cachedir, datadir, spkmap, sentfilt, context=50):
    isTrain = partition == 'train'
    if isTrain:
        rsr_ds = RandomFramesBatchFromCacheRsr2015(cachedir, context)
        rsr_ds = PrefetchDataZMQ(rsr_ds, min(8, multiprocessing.cpu_count()))
        print("Using", min(8, multiprocessing.cpu_count()), "threads")
    else:
        rsr_ds = WholeUtteranceAsBatchRsr2015(datadir, partition, spkmap, context=context, shuffle=isTrain, sentfilt=sentfilt)
    print(partition, isTrain, rsr_ds.size())
    return rsr_ds, rsr_ds.size()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=int, help='n context frms', default=50)
    parser.add_argument('--cachedir', help='dir to cache', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/trn_cache/context_50frms/')
    parser.add_argument('--datadir', help='dir to data', default='/data/sls/scratch/skoppula/kaldi-rsr/numpy/')
    parser.add_argument('--sentfilt', type=int, help='dir to data', default=None)
    parser.add_argument('--spkmap', help='dir to spk mappings', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle')
    args = parser.parse_args()

    train_dataflow, n_batches_trn = create_dataflow('train', args.cachedir, args.datadir, args.spkmap, args.sentfilt, context=args.context)
    val_dataflow, n_batches_val = create_dataflow('val', args.cachedir, args.datadir, args.spkmap, args.sentfilt, context=args.context)

    print("Using GPU:", os.environ['CUDA_VISIBLE_DEVICES'])

    logger.info("{} utterances per val epoch".format(n_batches_val))
    logger.info("Using host: {}".format(socket.gethostname()))

    model = Model()

    out_dir = '_'.join(['sentfilt' + str(args.sentfilt), 'fcn2', 'bn','w' + str(BITW), 'a' + str(BITA),'first_last_not_q_relu'])
    logger.set_logger_dir(os.path.join('train_log', out_dir), action='k')

    callbacks=[
        ModelSaver(),
        MinSaver('val-error-top1'),
        InferenceRunner(val_dataflow, [ScalarStats('cost'), ClassificationError('wrong-top1', 'val-error-top1'), ClassificationError('utt-wrong', 'val-utt-error')])
    ]

    config = TrainConfig(
        model=model,
        dataflow=train_dataflow,
        callbacks=callbacks,
        max_epoch=100,
        steps_per_epoch=n_batches_trn,
        nr_tower=max(get_nr_gpu(), 1),
    )

    if os.environ['CUDA_VISIBLE_DEVICES']:
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer([os.environ['CUDA_VISIBLE_DEVICES']], ps_device='gpu'))
    else:
        launch_train_with_config(config, SimpleTrainer())
