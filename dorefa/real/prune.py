import argparse

import tensorpack as tp
import math
import socket
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers.rsr_run import Model
from drf_run import *
from helpers.rsr_run import create_dataflow
from helpers.rsr_run import net_fn_map
from helpers.rsr2015 import *
from helpers.helpers import *
from helpers.baselinearchs import *
from tensorpack import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.varmanip import *
import os
from datetime import datetime

def avg(w): return sum(w.flatten())/len(w.flatten())

def prune(x, prune_rate=0.10):
    print("Pruning", x)
    a = np.copy(x)
    nonzero_vals = a[np.nonzero(a)]
    sorted_vals = np.sort(np.abs(nonzero_vals).flatten())
    split_idx = int(prune_rate * len(sorted_vals))
    thres = sorted_vals[split_idx]
    a[np.abs(a) < thres] = 0
    return a

def is_extra_var(var):
    return 'Adam' in var or 'global_step' == var or 'EMA' in var or var.startswith('beta') and var.endswith('_power') or 'bits_for_maxval_var' in var

def get_new_var_dict(var_dict, prune_rate=0.1, verbose=False):
    new_var_dict = {}
    ema_stats = set(); bn_vars = set(); kernel_vars = set(); bias_vars = set()
    print
    
    for var in var_dict:
        if is_extra_var(var):
            new_var_dict[var] = var_dict[var]
        elif 'bn' in var:
            bn_vars.add(var)
        elif var.endswith('/W') or var.endswith('/depthwise_weights'):
            kernel_vars.add(var)
        elif var.endswith('/b') or var.endswith('/biases'):
            bias_vars.add(var)
        else:
            print("Couldn't classify", var)

    for i,var in enumerate(kernel_vars):
        if verbose: print(var)
        new_var = prune(var_dict[var], prune_rate)
        new_var_dict[var] = new_var

    for i,var in enumerate(bn_vars):
        if verbose: print(var)
        new_var = prune(var_dict[var], prune_rate)
        new_var_dict[var] = new_var

    for i,var in enumerate(bias_vars):
        new_var_dict[var] = prune(var_dict[var], prune_rate)
    
    return new_var_dict, kernel_vars, bn_vars, bias_vars

def get_sparsity(var_dict, interesting_vars = None):
    zero_count = []
    totals = []
    if not interesting_vars: interesting_vars = var_dict.keys()
    for name in interesting_vars:
        if is_extra_var(name): continue
        weight = var_dict[name]
        zero_count.append(np.count_nonzero(weight==0))
        totals.append(np.prod(weight.shape))
    return float(sum(zero_count))/float(sum(totals)), zero_count, totals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='fcn2')
    parser.add_argument('--bita', type=int, help='bita', default=32)
    parser.add_argument('--bitw', type=int, help='bitw', default=32)
    parser.add_argument('--quant_ends', type=str2bool, nargs='?', const=True, default="n")
    parser.add_argument('--dorefa', type=str2bool, nargs='?', const=True, default="n")
    args = parser.parse_args()

    datadir = '/data/sls/scratch/skoppula/kaldi-rsr/numpy/'
    spkmap = '/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle'
    cachedir = '/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/trn_cache/context_50frms/'
    n_spks = get_n_spks(spkmap)
    
    if args.dorefa:
        ckpt_path = '/data/sls/u/meng/skanda/home/thesis/dorefa/real/train_log/'
        if args.quant_ends:
            ckpt_path += '{}_w_{}_a_{}_quant_ends_True_preload/checkpoint'.format(args.model, args.bitw, args.bita)
        else:
            ckpt_path += '{}_w_{}_a_{}_quant_ends_False/checkpoint'.format(args.model, args.bitw, args.bita)
    else:
        ckpt_path = '/data/sls/u/meng/skanda/home/thesis/manfxpt/models/'
        ckpt_path += 'sentfiltNone_' + args.model + '_bnTrue_regTrue_noLRSchedule/checkpoint'
    assert os.path.isfile(ckpt_path) 
    
    outdir=os.path.join('pruned_models', '_'.join([str(x) for x in [args.model, args.bitw, args.bita, args.quant_ends, args.dorefa]]))
    print("Outputting to outdir", outdir)
    logger.set_logger_dir(outdir, action='k')

    val_dataflow, n_batches_val = create_dataflow('val', cachedir, datadir, spkmap, None, context=50)

    logger.info("{} utterances per val epoch".format(n_batches_val))
    logger.info("Using host: {}".format(socket.gethostname()))

    net_fn_map = {'fcn1':fcn1_net, 'fcn2':fcn2_net, 'cnn':cnn_net, 'maxout2':maxout2_net, 'maxout1':maxout1_net, 'lcn':lcn_net, 'dsc1':dsc1_net, 'dsc2':dsc2_net}
    

    prune_rates = [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.25, 0.5]
    errors = []
    sparsities = []

    for i, prune_rate in enumerate(prune_rates):
        logger.info("On prune rate {}".format(prune_rate))
        
        val_dataflow, n_batches_val = create_dataflow('val', None, datadir, spkmap, None, 50)
        val_generator = val_dataflow.get_data()
        
        var_dict = load_chkpt_vars(ckpt_path)
        new_var_dict, _, _, _ = get_new_var_dict(var_dict, prune_rate)
        sparsities.append(get_sparsity(new_var_dict)[0])

        if args.dorefa:
            model = DoReFaModel(args.bitw, args.bita, net_fn_map[args.model], args.quant_ends)
        else:
            model = Model(n_spks, net_fn_map[args.model], bn=True, reg=True, n_context=50)

        config = PredictConfig(
                model=model,
                session_init=DictRestore(new_var_dict),
                input_names=['input', 'label'],
                output_names=['utt-wrong']
        )
        predictor = OfflinePredictor(config)

        rc = tp.utils.stats.RatioCounter()
        for i in range(n_batches_val):
            x,y = next(val_generator)
            outputs, = predictor([x,y])
            rc.feed(outputs,1)
            if i % 100 == 0:
                print("On",i,"of",n_batches_val,"error:", rc.ratio)
            if i == 700: break
        logger.info("error: {}".format(rc.ratio))
        errors.append(rc.ratio[0])
    logger.info("Errors:" + str(errors))
    logger.info("Sparsities:" + str(sparsities))
