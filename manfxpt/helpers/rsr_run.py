import numpy as np
import multiprocessing
import argparse
import socket
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import dataset
from tensorpack.tfutils.common import get_op_or_tensor_by_name

from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.utils.develop import log_deprecated

from custom_layers_with_mult_tracking import *
from rsr2015 import *
from helpers import get_tensors_from_graph
from helpers import DumpTensorsOnce
from helpers import str2bool
from quantization import *
from baselinearchs import *

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

net_fn_map = {'fcn1':fcn1_net, 'fcn2':fcn2_net, 'cnn':cnn_net, 'maxout2':maxout2_net, 'maxout1':maxout1_net, 'lcn':lcn_net, 'dsc1':dsc1_net, 'dsc2':dsc2_net}

class Model(ModelDesc):

    def __init__(self, n_spks, net_fn, bn=True, reg=True, n_context=50, qtype=None, w_bits=8, a_bits=8, overflow_rate=0.01):
        super(Model, self).__init__()
        
        self.n_spks = n_spks
        self.n_context = n_context
        self.network_complexity = {'mults':0, 'weights':0}
        self.ternary_weight_tensors = []
        self.net_fn = net_fn
        
        self.batchnorm = bn
        self.regularize = reg
        
        self.qtype = qtype; self.w_bits = w_bits; self.a_bits = a_bits; self.overflow_rate = overflow_rate; self.qtype = qtype


    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, self.n_context*20], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        inp, label = inputs

        self.nl = BNReLU if self.batchnorm else tf.nn.relu
        if self.net_fn == maxout1_net or self.net_fn == maxout2_net:
            self.nl = bnorm if self.batchnorm else tf.identity
        else:
            self.nl = BNReLU if self.batchnorm else tf.nn.relu

        if self.qtype:
            self.nl = get_quantized_nl(self.nl, self.a_bits, self.overflow_rate, self.qtype)
        
        with argscope([Conv2DWithTrackedMults, BatchNorm], data_format='NHWC'), \
                argscope([Conv2DWithTrackedMults], W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2DWithTrackedMults], network_complexity=self.network_complexity), \
                argscope([FullyConnectedWithTrackedMults], network_complexity=self.network_complexity), \
                argscope([DepthwiseSeparableConvWithTrackedMults], network_complexity=self.network_complexity):
            l = self.net_fn(inp, self.nl, self.n_context)
        
        logits = FullyConnectedWithTrackedMults('last_linear', l, out_dim=self.n_spks, nl=tf.identity, network_complexity=self.network_complexity)

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

        if self.qtype:
            with tf.name_scope('quantization-summary'):
                def fn(name): return 'bits_for_maxval' in name and not('Initializer' in name or 'var' in name or 'Assign' in name)
                def fn2(name): return 'quantized_output' in name
                self.bits_for_maxval_tensors = get_tensors_from_graph(tf.get_default_graph(), fn)
                self.quantized_outputs = get_tensors_from_graph(tf.get_default_graph(), fn2)
                print("Adding quantization bits for max val to summary:", self.bits_for_maxval_tensors)
                for tensor in self.bits_for_maxval_tensors:
                    add_tensor_summary(tensor, ['rms'])
                print("Adding quantized output activations to summary:", self.quantized_outputs)
                for tensor in self.quantized_outputs:
                    add_tensor_summary(tensor, ['rms', 'histogram'])
            
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
        # lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.AdamOptimizer()
        return opt

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
    parser.add_argument('--model_name', help='output folder name', default='fcn')
    parser.add_argument('--context', type=int, help='n context frms', default=50)
    parser.add_argument('--load_ckpt', help='ckpt load', default=None)
    parser.add_argument('--notes', help='', default=None)
    parser.add_argument('--outdir', help='alternative outdir', default=None)
    parser.add_argument('--bn', type=str2bool, nargs='?', const=True, help='use batchnorm', default="y")
    parser.add_argument('--reg', type=str2bool, nargs='?', const=True, help='regularize', default="n")

    parser.add_argument('--cachedir', help='dir to cache', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/trn_cache/context_50frms/')
    parser.add_argument('--datadir', help='dir to data', default='/data/sls/scratch/skoppula/kaldi-rsr/numpy/')
    parser.add_argument('--sentfilt', type=int, help='dir to data', default=None)
    parser.add_argument('--spkmap', help='dir to spk mappings', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle')
    args = parser.parse_args()


    print("Using GPU:", os.environ['CUDA_VISIBLE_DEVICES'])

    if args.outdir is None:
        if args.notes:
            out_dir = '_'.join(['sentfilt' + str(args.sentfilt), args.model_name, 'bn' + str(args.bn), 'reg' + str(args.reg), args.notes])
        else:
            out_dir = '_'.join(['sentfilt' + str(args.sentfilt), args.model_name, 'bn' + str(args.bn), 'reg' + str(args.reg)])
        if args.load_ckpt:
            out_dir += '_preload'
    else:
        out_dir = args.outdir
    logger.set_logger_dir(os.path.join('train_log', out_dir), action='k')
    batch_size = 512;

    logger.info("Using sentence filter: {}".format(str(args.sentfilt)))
    train_dataflow, n_batches_trn = create_dataflow('train', args.cachedir, args.datadir, args.spkmap, args.sentfilt, context=args.context)
    val_dataflow, n_batches_val = create_dataflow('val', args.cachedir, args.datadir, args.spkmap, args.sentfilt, context=args.context)
    logger.info("{} utterances per val epoch".format(n_batches_val))
    logger.info("Using host: {}".format(socket.gethostname()))

    n_spks = get_n_spks(args.spkmap)
    logger.info("Using {} speaker".format(n_spks))

    model = Model(n_spks, net_fn_map[args.model_name], args.bn, args.reg, args.context)

    callbacks=[
        ModelSaver(),
        MinSaver('val-error-top1'),
        InferenceRunner(val_dataflow, [ScalarStats('cost'), ClassificationError('wrong-top1', 'val-error-top1'), ClassificationError('utt-wrong', 'val-utt-error')])
        # ScheduledHyperParamSetter('learning_rate', [(3, 0.1), (20, 0.01), (30, 0.001), (50, 0.0002)])
    ]

    config = TrainConfig(
        model=model,
        dataflow=train_dataflow,
        callbacks=callbacks,
        max_epoch=100,
        nr_tower=max(get_nr_gpu(), 1),
        steps_per_epoch=n_batches_trn,
        session_init=SaverRestore(args.load_ckpt) if args.load_ckpt else None
    )

    if os.environ['CUDA_VISIBLE_DEVICES']:
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer([os.environ['CUDA_VISIBLE_DEVICES']], ps_device='gpu'))
    else:
        launch_train_with_config(config, SimpleTrainer())

