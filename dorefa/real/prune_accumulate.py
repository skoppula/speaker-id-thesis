import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

plt.style.use('fivethirtyeight')

models = ['lcn', 'fcn1', 'fcn2', 'cnn']
qend="False"

prune_rates = [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.25, 0.5]
print(prune_rates)

for dorefa, bitw in [("True", 4), ("True", 8), ("True", 16), ("False", 32)]:
    errs = np.zeros((len(models), len(prune_rates)))
    sps = np.zeros((len(models), len(prune_rates)))
    for j, model in enumerate(models):
        logpath='/data/sls/u/meng/skanda/home/thesis/dorefa/real/'
        logpath+='pruned_models/{}_{}_32_{}_{}/'.format(model,bitw,qend,dorefa)
        logpath+='log.log'

        with open(logpath,'r') as f:
            for line in f:
                if 'Errors:[' in line:
                    st = line.split('Errors:')[1]
                    errs[j] = np.fromstring(st[1:-1], dtype=float, sep=', ')
                elif 'Sparsities:[' in line:
                    st = line.split('Sparsities:')[1]
                    sps[j] = np.fromstring(st[1:-1], dtype=float, sep=', ')
                
    points = np.array(errs)
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('Speaker ID Error')
    for i, name in enumerate(['Locally Connected', 'Fully Connected (Small)', 'Fully Connected (Large)', 'Convolutional']):
        if i < points.shape[0]:
            ax1.plot(prune_rates, points[i,:], '-o', label=name)
        print(dorefa, name, bitw, errs[i])
    ax1.legend(loc=2)
    title = "Error vs. Induced Sparsity"
    if dorefa == "True":
        title += ", FxPt Quant. {}-Bit".format(bitw)
        filename="dorefa_{}".format(bitw)
    else:
        title += ", Full-Precision"
        filename="baseline"
    plt.title(title)
    plt.savefig('figs/pruned/%s.png' % (filename,))
    print(dorefa, bitw, sps[:,1])
