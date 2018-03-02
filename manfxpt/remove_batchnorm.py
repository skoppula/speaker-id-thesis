import tensorpack as tp
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from helpers.rsr_run import Model
from helpers.rsr_run import create_dataflow
from helpers.rsr_run import net_fn_map
from helpers.rsr2015 import *
from tensorpack import *
from tensorpack.tfutils.varmanip import *
from tensorpack.utils.gpu import get_nr_gpu
from datetime import datetime
import os
import argparse

def fuse_bn_params(kernel, bias, beta, gamma, mean_ema, var_ema):
    kernel,bias = kernel.astype(np.float64), bias.astype(np.float64)
    beta,gamma = beta.astype(np.float64), gamma.astype(np.float64)
    mean_ema,var_ema = mean_ema.astype(np.float64), var_ema.astype(np.float64)
    scale = gamma/np.sqrt(var_ema+1e-5)
    new_kernel = kernel*scale
    new_bias = beta-scale*(bias+mean_ema)
    return new_kernel.astype(np.float32), new_bias.astype(np.float32)

def reorder(names):
    names = sorted(names)
    for i, name in enumerate(names):
        if '/depthwise_weights' in name:
            var = names.pop(i)
            names.insert(0, var)
            break
    for i, name in enumerate(names):
        if '/biases' in name:
            var = names.pop(i)
            names.insert(1, var)
            break
    return names

def fuse_bn_layer(var_dict, layer_var_names):
    layer_vars = reorder(layer_var_names)
    print(layer_vars)
    layer_vars = [var_dict[var] for var in layer_vars]
    
    if len(layer_vars) == 6:
        return fuse_bn_params(*layer_vars)
    else: assert False

def group_bn_vars(var_dict):
    all_bn_vars = set([var for var in var_dict if '/bn/' in var and 'Adam' not in var])
    base_scopes_for_bn = set([var.split('/bn/')[0] for var in all_bn_vars])
    groups = {scope:[var for var in all_bn_vars if scope in var] for scope in base_scopes_for_bn}
    for scope in groups:
        for var in var_dict:
            if 'Adam' in var or '/bn/' in var: continue
            if scope in var and (var.endswith('/W') or var.endswith('/b') or var.endswith('/depthwise_weights') or var.endswith('/biases')):
                groups[scope].insert(0, var)
    return groups

def fuse_bn_layers(var_dict):
    groups = group_bn_vars(var_dict)
    print(groups)
    new_var_dict = {}
    new_groups = {}
    replaced_vars = set()
    for layer in groups:
        new_w, new_b = fuse_bn_layer(var_dict, groups[layer])
        for var_name in groups[layer]:
            if var_name.endswith('/W') or var_name.endswith('/depthwise_weights'):
                new_var_dict[var_name] = new_w
                replaced_vars.add(var_name)
            elif var_name.endswith('/b') or var_name.endswith('/biases'):
                new_var_dict[var_name] = new_b
                replaced_vars.add(var_name)
        print("processing layer", layer, new_w.shape, new_b.shape)
    for var in var_dict:
        if '/bn/' in var or var in replaced_vars or 'Adam' in var: continue
        new_var_dict[var] = var_dict[var]
    return new_var_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='output folder name', default='lcn')
    parser.add_argument('--load_ckpt', help='ckpt load', default='/data/sls/u/meng/skanda/home/thesis/manfxpt/models/sentfiltNone_lcn_bnTrue_regTrue_noLRSchedule/checkpoint')
    parser.add_argument('--datadir', default='/data/sls/scratch/skoppula/kaldi-rsr/numpy/')
    parser.add_argument('--spkmap', default='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle')
    args = parser.parse_args()

    spkmap = args.spkmap; datadir=args.datadir; model_name = args.model_name; bn = True; ckpt_path = args.load_ckpt;

    outdir=os.path.join('no_bn_models', '_'.join([str(x) for x in [model_name]]))
    logger.info("Outputting to outdir {}".format(outdir))
    logger.info("Using checkpoint {}".format(ckpt_path))
    logger.set_logger_dir(outdir, action='k')
    context=50
    n_spks = get_n_spks(spkmap)
    
    model = Model(n_spks, net_fn_map[model_name], bn=False, reg=True, n_context=context, qtype=None)
    var_dict = load_chkpt_vars(ckpt_path)
    new_var_dict = fuse_bn_layers(var_dict)

    with TowerContext('', is_training=False):
        input = PlaceholderInput()
        input.setup(model.get_inputs_desc())
        model.build_graph(*input.get_input_tensors())

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = sessinit.DictRestore(new_var_dict)
        sess.run(tf.global_variables_initializer())
        init.init(sess)

        ms = ModelSaver(checkpoint_dir=outdir)
        ms._setup_graph()
        time = datetime.now().strftime('%m%d-%H%M%S')
        ms.saver.export_meta_graph(os.path.join(ms.checkpoint_dir, 'graph-{}.meta'.format(time)), collection_list=tf.get_default_graph().get_all_collection_keys())
        ms.saver.save(sess, ms.path, global_step=0, write_meta_graph=False)

        np.savez_compressed(os.path.join(outdir, 'params.npz'), **new_var_dict)

    val_dataflow, n_batches_val = create_dataflow('val', None, datadir, spkmap, None, context)
    val_generator = val_dataflow.get_data()
    config = PredictConfig(
            model=model,
            session_init=SaverRestore(os.path.join(outdir, 'checkpoint')),
            input_names=['input', 'label'],
            output_names=['utt-wrong']
    )
    predictor = OfflinePredictor(config)

    rc = tp.utils.stats.RatioCounter()
    n_steps_inference=1000
    for i in range(n_steps_inference):
        x,y = next(val_generator)
        outputs = predictor([x,y])[0]
        rc.feed(outputs,1)
        if i % 100 == 0:
            print("On",i,"of",n_steps_inference, "error:", rc.ratio)
    logger.info("Final cumulative utt accuracy after {} steps: {}".format(n_steps_inference, rc.ratio))

