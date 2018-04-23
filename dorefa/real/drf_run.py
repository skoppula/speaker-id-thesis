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

class DoReFaModel(ModelDesc):
    def __init__(self, bitw, bita, net_fn, quant_ends):
        super(DoReFaModel, self).__init__()
        
        self.n_spks = 255
        self.n_context = 50
        self.net_fn = net_fn
        self.network_complexity = {'mults':0, 'weights':0}
        self.bitw, self.bita = bitw, bita
        self.quant_ends = quant_ends
        
        
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, self.n_context*20], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        inp, label = inputs
        is_training = get_current_tower_context().is_training

        fw, fa = get_dorefa(self.bitw, self.bita)

        def binarize_weight(v):
            name = v.op.name
            if not (name.endswith('W') or name.endswith('b')):
                logger.info("Not quantizing {}".format(name))
                return v
            elif not self.quant_ends and 'conv0' in name:
                logger.info("Not quantizing {}".format(name))
                return v
            elif not self.quant_ends and 'last_linear' in name:
                logger.info("Not quantizing {}".format(name))
                return v
            elif not self.quant_ends and (self.net_fn == fcn1_net or self.net_fn == fcn2_net) and 'linear0' in name:
                logger.info("Not quantizing {}".format(name))
                return v
            else:
                logger.info("Quantizing weight {}".format(name))
                return fw(v)

        def nonlin(x, name="activate"):
            if self.bita == 32:
                return fa(tf.nn.relu(BNWithTrackedMults(x)))
            else:
                return fa(tf.clip_by_value(BNWithTrackedMults(x), 0.0, 1.0))

        with remap_variables(binarize_weight), \
                argscope([FullyConnectedWithTrackedMults], network_complexity=self.network_complexity), \
                argscope([Conv2DWithTrackedMults], network_complexity=self.network_complexity), \
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
            logger.info("Adding activation tensors to summary: {}".format(tensors))
            for tensor in tensors:
                add_tensor_summary(tensor, ['rms', 'histogram'])

        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(), 480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(wd_cost)
        self.cost = tf.add_n([cost, wd_cost], name='cost')

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
        rsr_ds = PrefetchDataZMQ(rsr_ds, min(5, multiprocessing.cpu_count()))
        logger.info("Using {} threads".format(min(5, multiprocessing.cpu_count())))
    else:
        rsr_ds = WholeUtteranceAsBatchRsr2015(datadir, partition, spkmap, context=context, shuffle=isTrain, sentfilt=sentfilt)
    return rsr_ds, rsr_ds.size()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='output folder name', default='fcn2')
    parser.add_argument('--cachedir', help='dir to cache', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/trn_cache/context_50frms/')
    parser.add_argument('--datadir', help='dir to data', default='/data/sls/scratch/skoppula/kaldi-rsr/numpy/')
    parser.add_argument('--spkmap', help='dir to spk mappings', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle')
    parser.add_argument('--load_ckpt', help='load ckpt', default=None)
    parser.add_argument('--bita', type=int, help='bita', default=32)
    parser.add_argument('--bitw', type=int, help='bitw', default=32)
    parser.add_argument('--quant_ends', type=str2bool, nargs='?', const=True, help='quantize first and last layers', default="n")
    args = parser.parse_args()

    out_dir = '_'.join([args.model_name, 'w', str(args.bitw), 'a', str(args.bita),'quant_ends', str(args.quant_ends)])
    if args.load_ckpt:
       out_dir += '_preload' 
    logger.set_logger_dir(os.path.join('train_log', out_dir), action='k')

    train_dataflow, n_batches_trn = create_dataflow('train', args.cachedir, args.datadir, args.spkmap, None, context=50)
    val_dataflow, n_batches_val = create_dataflow('val', args.cachedir, args.datadir, args.spkmap, None, context=50)

    logger.info("{} utterances per val epoch".format(n_batches_val))
    logger.info("Using host: {}".format(socket.gethostname()))

    net_fn_map = {'fcn1':fcn1_net, 'fcn2':fcn2_net, 'cnn':cnn_net, 'maxout2':maxout2_net, 'maxout1':maxout1_net, 'lcn':lcn_net, 'dsc1':dsc1_net, 'dsc2':dsc2_net}
    model = DoReFaModel(args.bitw, args.bita, net_fn_map[args.model_name], args.quant_ends)

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
        session_init=SaverRestore(args.load_ckpt) if args.load_ckpt else None
    )

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logger.info("Using GPU: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer([os.environ['CUDA_VISIBLE_DEVICES']], ps_device='gpu'))
    else:
        logger.info("Using no GPU")
        launch_train_with_config(config, SimpleTrainer())
