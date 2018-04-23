import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
a = np.loadtxt('/tmp/sparsitycounter')
fig, ax = plt.subplots()
ax.stackplot(list(range(a.shape[1])), a[0], a[1], a[2], labels=["Sparsity", "Negative Weight Wp", "Positive Weight Wn"])
ax.legend(loc=2)
plt.savefig(sys.argv[1])
