import matplotlib.pyplot as plt
import numpy as np

filepath = 'workdir/output-data/HeNe-cycles0-step0.0-samples100000.txt'
cols = (0, 1, 2)
data = np.loadtxt(filepath, delimiter='\t', skiprows=1, usecols=cols, encoding='utf-8')
data = data[:, 2]
xs = np.arange(1, data.size + 1, step=1)

plt.plot(data)
plt.show()
