import pandas as pd
import matplotlib.pyplot as plt

import rapp.constants as ct

filepath = 'workdir/output-data/2023-12-07-HeNe-noise-cycles0-step10-samples100000.txt'
data = pd.read_csv(
    filepath,
    sep=r"\s+", skip_blank_lines=True, comment='#', header=0, usecols=(0, 1, 2),
    encoding=ct.ENCONDIG
)


data_ch0 = data['CH0']
data_ch1 = data['CH1']

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

ax0.plot(data_ch0)
ax1.plot(data_ch1)

plt.show()
