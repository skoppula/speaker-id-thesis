import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

qend = sys.argv[1] == "True"
act= sys.argv[2] == "True"

plt.style.use('fivethirtyeight')

with open('/tmp/plot.txt') as f:
    lines = f.readlines()

points = []
for line in lines:
    if not line.strip(): continue
    lst = [float(x) for x in line.split(' ')]
    if len(lst) < 5:
        print("problem with line", lst)
        continue
    points.append(lst)
points = np.array(points)
x = np.array([2,4,8,16,32])

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Bitwidth of Weights')
ax1.set_ylabel('Speaker ID Error')
print(points.shape, points)
for i, name in enumerate(['Locally Connected', 'Fully Connected (Small)', 'Fully Connected (Large)', 'Convolutional']):
    if i < points.shape[0]:
        ax1.plot(x, points[i,:], '-o', label=name)
ax1.legend(loc=2)
title = "0-1 FxPt Quantization: Error vs. Bitwidth"
title += ", (All Layers" if qend else ", (Middle Layers"
title += ", Weights & Activations Quant.)" if qend else ", Weights Quant.)"
plt.title(title)
plt.savefig('figs/dorefa_bitwidths_qend%s_act%s.png' % (qend,act))
