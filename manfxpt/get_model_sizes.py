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

model_name = 'fcn2'
spkmap='/data/sls/scratch/skoppula/backup-exps/rsr-experiments/create_rsr_data_cache/generator_full_dataset/spk_mappings.pickle'
context=50
n_spks = get_n_spks(spkmap)



model_names = ['fcn1','fcn2','cnn', 'lcn', 'maxout1','maxout2','dsc1','dsc2']
dicts = []
for model_name in model_names:
    tf.reset_default_graph()

    print("No batch norm:")
    model = Model(n_spks, net_fn_map[model_name], bn=False, reg=True, n_context=context, qtype=None)
    with TowerContext('', is_training=False):
        input = PlaceholderInput()
        input.setup(model.get_inputs_desc())
        model.build_graph(*input.get_input_tensors())
        # dicts[model_name + '_nobn'] = model.network_complexity
        dicts.append(model.network_complexity)
        

    tf.reset_default_graph()

    print("With batch norm:")
    model = Model(n_spks, net_fn_map[model_name], bn=True, reg=True, n_context=context, qtype=None)
    with TowerContext('', is_training=False):
        input = PlaceholderInput()
        input.setup(model.get_inputs_desc())
        model.build_graph(*input.get_input_tensors())
        dicts.append(model.network_complexity)

for i, d in enumerate(dicts):
    if i % 2 == 1: continue
    print(model_names[i/2], d)

for i, d in enumerate(dicts):
    if i % 2 == 0: continue
    print(model_names[i/2], d)
