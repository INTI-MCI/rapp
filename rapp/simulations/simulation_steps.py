import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal import signal


logger = logging.getLogger(__name__)


def run(folder, method=None, samples=None, step=None, reps=None, cycles=None, show=False):
    print("")
    logger.info("SIMULATION PROCESS...")

    cycles = 0.15
    fc = simulator.samples_per_cycle(step=0.5)
    noise = (0, 0.04)
    mu, sigma = noise
    bits = 6
    A = 1.7

    f, axs = plt.subplots(4, 1, figsize=(4, 10), sharey=True)

    # Pure signal
    xs, ys = signal.harmonic(A=A, cycles=cycles, fc=fc, all_positive=True)
    xs = np.rad2deg(xs)
    axs[0].plot(xs, ys, 'o-', color='k', ms=2, mfc='None')
    axs[0].set_ylabel(ct.LABEL_VOLTAGE)
    axs[0].set_xlabel(ct.LABEL_DEGREE)

    # Noisy signal
    xs, ys = signal.harmonic(A=A, cycles=cycles, fc=fc, noise=noise, all_positive=True)
    xs = np.rad2deg(xs)
    axs[1].plot(xs, ys, 'o-', color='k', ms=2, mfc='None', label="σ={}".format(sigma))
    axs[1].set_ylabel(ct.LABEL_VOLTAGE)
    axs[1].set_xlabel(ct.LABEL_DEGREE)
    axs[1].legend(loc='lower right', prop={'family': 'monaco', 'size': 12})

    # Quantized signal
    xs, ys = signal.harmonic(A=A, cycles=cycles, fc=fc, noise=noise, bits=bits, all_positive=True)
    xs = np.rad2deg(xs)

    label = "σ={}\nbits={}".format(sigma, bits)
    axs[2].plot(xs, ys, 'o-', color='k', ms=2, mfc='None', label=label)
    axs[2].set_ylabel(ct.LABEL_VOLTAGE)
    axs[2].set_xlabel(ct.LABEL_DEGREE)
    axs[2].legend(loc='lower right', prop={'family': 'monaco', 'size': 12})

    # Quantized signal + 50 samples
    fa = 50
    label = "σ={}\nbits={}\nmuestras={}".format(sigma, bits, fa)

    xs, ys = signal.harmonic(
        A=A, cycles=cycles, fc=fc, fa=fa, noise=noise, bits=bits, all_positive=True)

    data = np.array([xs, ys]).T
    data = pd.DataFrame(data=data, columns=["ANGLE", "CH0"])
    data = data.groupby(['ANGLE'], as_index=False).agg({'CH0': ['mean', 'std']})
    xs = np.array(data['ANGLE'])
    ys = np.array(data['CH0']['mean'])

    xs = np.rad2deg(xs)

    axs[3].plot(xs, ys, 'o-', color='k', ms=2, mfc='None', label=label)
    axs[3].set_ylabel(ct.LABEL_VOLTAGE)
    axs[3].set_xlabel(ct.LABEL_DEGREE)
    axs[3].legend(loc='lower right', prop={'family': 'monaco', 'size': 12})

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    f.savefig(os.path.join(folder, 'sim_steps.png'))

    if show:
        plt.show()

    plt.close()

    logger.info("Done.")
