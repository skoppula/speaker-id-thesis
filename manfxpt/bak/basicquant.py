import tensorpack as tp
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import socket

from helpers.helpers import str2bool
from helpers.rsr_run import Model
from helpers.rsr_run import create_dataflow
from helpers.rsr_run import net_fn_map
from helpers.rsr2015 import *
from tensorpack import *
from tensorpack.utils.gpu import get_nr_gpu
import os

def compute_nbits_for_maxval(inp, overflow_rate):
    sorted_vals = np.sort(np.abs(inp).flatten())[::-1]
    split_idx = int(overflow_rate * len(sorted_vals))
    v = sorted_vals[split_idx]
    return math.ceil(math.log(v+1e-12, 2))

def linear_quantize(v, w_bits, bits_per_delta):
    bound = math.pow(2.0, w_bits-1)
    min_val, max_val = - bound, bound - 1
    delta = math.pow(2.0, bits_per_delta)
    quantized = np.round(v / delta)
    clipped = np.clip(quantized, min_val, max_val)
    return clipped*delta

def quantize_var(v, n_bits, overflow_rate=0.01):
    # +1 bc we get extra bit by taking abs val and seperating pos/neg
    bits_per_delta = compute_nbits_for_maxval(v, overflow_rate) + 1. - n_bits
    return linear_quantize(v, n_bits, bits_per_delta)

def get_new_var_dict(ckpt_path, w_bits=32, bn_bits=32, bias_bits=32):
    var_dict = tp.tfutils.varmanip.load_chkpt_vars(ckpt_path)
    new_var_dict = {}
    ema_stats = set(); bn_vars = set(); kernel_vars = set(); bias_vars = set()
    for var in var_dict:
        if 'Adam' in var or 'global_step' == var or 'bits_for_maxval_var' in var:
            new_var_dict[var] = var_dict[var]
            continue
        elif var.startswith('beta') and var.endswith('_power'):
            new_var_dict[var] = var_dict[var]
            continue
        elif var.startswith('EMA/'):
            new_var_dict[var] = var_dict[var]
            ema_stats.add(var);
        elif '/bn/' in var:
            bn_vars.add(var);
        elif var.endswith('/W') or var.endswith('depthwise_weights'):
            kernel_vars.add(var)
        elif var.endswith('/b') or var.endswith('/biases'):
            bias_vars.add(var)
        elif 'bn' in var:
            bn_vars.add(var)
        else:
            print("Couldn't classify", var)

    for i,var in enumerate(kernel_vars):
        new_var_dict[var] = quantize_var(var_dict[var], w_bits, overflow_rate=0.01)

    for i,var in enumerate(bn_vars):
        new_var_dict[var] = quantize_var(var_dict[var], bn_bits, overflow_rate=0.01)

    for i,var in enumerate(bias_vars):
        new_var_dict[var] = quantize_var(var_dict[var], bias_bits, overflow_rate=0.01)
    
    return new_var_dict

def print_vars_in_ckpt(ckpt_path='/data/sls/u/meng/skanda/home/thesis/manfxpt/train_log/fcn2_bn_True_reg_False_quant/model-4684087'):
    var_dict = tp.tfutils.varmanip.load_chkpt_vars(ckpt_path)
    for var in var_dict.keys():
        if 'maxval' in var:
            print var, var_dict[var]
            
def plot_quant_hist(var_pairs, var_names):
    fig = plt.figure(1, figsize=(8,8*len(var_pairs)))
    for i, var in enumerate(var_names):
        plt.subplot(len(var_pairs)*2,2,2*i+1)
        plt.hist(var_pairs[i][0].flatten(),bins=100); plt.title(var);
        plt.grid(True)
        plt.subplot(len(var_pairs)*2,2,2*i+2)
        plt.hist(var_pairs[i][1].flatten(),bins=100); plt.title(var + '_quant');
        plt.grid(True)
    plt.show()
    plt.clf()

def plot_all_vars(kernel_vars, bn_vars, bias_vars):
    var_pairs = []
    for i,var in enumerate(kernel_vars):
        var_pairs.append((var_dict[var], new_var_dict[var]))
    plot_quant_hist(var_pairs, list(kernel_vars))

    var_pairs = []
    for i,var in enumerate(bn_vars):
        var_pairs.append((var_dict[var], new_var_dict[var]))
    plot_quant_hist(var_pairs, list(bn_vars))

    var_pairs = []
    for i,var in enumerate(bias_vars):
        var_pairs.append((var_dict[var], new_var_dict[var]))
    plot_quant_hist(var_pairs, list(bias_vars))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='output folder name', default='fcn2')
    parser.add_argument('--load_ckpt', help='ckpt load', default='/data/sls/u/meng/skanda/home/thesis/manfxpt/models/sentfiltNone_fcn2_bnTrue_regFalse_noLRSchedule/checkpoint')
    parser.add_argument('--cachedir', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/trn_cache/context_50frms/')
    parser.add_argument('--datadir', default='/data/sls/scratch/skoppula/kaldi-rsr/numpy/')
    parser.add_argument('--spkmap', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle')
    parser.add_argument('--bn', type=str2bool, nargs='?', const=True, help='use batchnorm', default="y")
    parser.add_argument('--reg', type=str2bool, nargs='?', const=True, help='regularize', default="n")
    parser.add_argument('--a_bits', type=int, default=32)
    parser.add_argument('--bias_bits', type=int, default=32)
    parser.add_argument('--w_bits', type=int, default=32)
    parser.add_argument('--bn_bits', type=int, default=32)
    parser.add_argument('--overflow_rate', type=float, default=0.01)
    parser.add_argument('--n_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--n_steps_inference', type=int, default=1000)
    args = parser.parse_args()

    a_bits = args.a_bits; w_bits = args.w_bits; bn_bits = args.bn_bits; bias_bits = args.bias_bits; overflow_rate = args.overflow_rate; n_steps_per_epoch = args.n_steps_per_epoch; n_steps_inference = args.n_steps_inference; spkmap = args.spkmap; datadir=args.datadir; cachedir = args.cachedir; model_name = args.model_name; bn = args.bn; reg = args.reg; ckpt_path = args.load_ckpt;

    outdir=os.path.join('train_log', '_'.join([str(x) for x in [model_name, 'bn',bn, 'reg',reg,'wbit',w_bits, 'abit', a_bits,'bnbit',bn_bits,'biasbit',bias_bits,'overflow',overflow_rate]]))
    logger.set_logger_dir(outdir, action='k')
    context=50

    train_dataflow, n_batches_trn = create_dataflow('train', cachedir, datadir, spkmap, None, context)
    val_dataflow, n_batches_val = create_dataflow('val', None, datadir, spkmap, None, context)
    n_spks = get_n_spks(spkmap)
    if not n_steps_per_epoch or n_steps_per_epoch == 'None': n_steps_per_epoch = n_batches_trn
    if not n_steps_inference or n_steps_inference == 'None': n_steps_per_inference = n_batches_val

    new_var_dict = get_new_var_dict(ckpt_path, w_bits, bn_bits, bias_bits)
    model = Model(n_spks, net_fn_map[model_name], bn=bn, reg=reg, n_context=context, qtype='linear', w_bits=w_bits, a_bits=a_bits, overflow_rate=overflow_rate)
    config = TrainConfig(
        model=model,
        dataflow=train_dataflow,
        callbacks=[ModelSaver(keep_checkpoint_every_n_hours=0.001)],
        max_epoch=2,
        nr_tower=max(get_nr_gpu(), 1),
        steps_per_epoch=n_steps_per_epoch,
        session_init=DictRestore(new_var_dict)
    )

    launch_train_with_config(config, SimpleTrainer())


    val_generator = val_dataflow.get_data()
    new_var_dict = get_new_var_dict(os.path.join(outdir, 'checkpoint'), w_bits, bn_bits, bias_bits)

    config = PredictConfig(
            model=model,
            session_init=DictRestore(new_var_dict),
            input_names=['input', 'label'],
            output_names=['utt-wrong']
    )
    predictor = OfflinePredictor(config)

    rc = tp.utils.stats.RatioCounter()
    for i in range(n_steps_inference):
        x,y = next(val_generator)
        outputs = predictor([x,y])[0]
        rc.feed(outputs,1)
        if i % 100 == 0:
            print("On",i,"of",n_steps_inference, "error:", rc.ratio)
    logger.info("Final cumulative utt accuracy after {} steps: {}".format(n_steps_inference, rc.ratio))

