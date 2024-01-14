import logging

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from rapp import constants as ct

logger = logging.getLogger(__name__)


def plot_drift(output_folder, show=False):
    print("")
    logger.info("PROCESSING LASER DRIFT...")

    reencendido = np.loadtxt(
        "data/old/continuous-range4V-584nm-16-reencendido-samples1M.txt",
        delimiter=' ', skiprows=1, usecols=(1, 2), encoding=ct.ENCONDIG)

    r0 = reencendido[:, 0]
    r1 = reencendido[:, 1]

    drift0 = r0[300000:]
    drift1 = r1[300000:]

    plt.figure()
    plt.plot(drift0)

    if show:
        plt.show()

    # plt.close()

    plt.figure()
    plt.plot(drift1)

    if show:
        plt.show()

    plt.close()

    plt.figure()
    fft_data0 = np.fft.fft(drift0)
    fft_data1 = np.fft.fft(drift1)
    plt.semilogy(np.abs(fft_data0))
    plt.semilogy(np.abs(fft_data1))

    if show:
        plt.show()

    plt.close()

    sf = 250000
    fc = np.array([100, 1000])
    w = fc / (sf/2)
    print(w)
    figure, ax = plt.subplots(len(fc), 2, figsize=(8, 4*len(fc)), sharex=False, sharey=False)
    figure.suptitle('Deriva filtrada')

    b1, a1 = signal.butter(3, w[0], btype='lowpass')
    filtered_drift0 = signal.filtfilt(b1, a1, drift0)
    filtered_drift1 = signal.filtfilt(b1, a1, drift1)

    print(filtered_drift0)
    print(filtered_drift1)

    ax[0, 0].plot(filtered_drift0)
    ax[0, 0].set_title('Canal 0, fc = {}'.format(fc[0]))
    ax[0, 1].plot(filtered_drift1)
    ax[0, 1].set_title('Canal 1, fc = {}'.format(fc[0]))

    b1, a1 = signal.butter(3, w[1], btype='lowpass')
    filtered_drift0 = signal.filtfilt(b1, a1, drift0)
    filtered_drift1 = signal.filtfilt(b1, a1, drift1)

    print(filtered_drift0)
    print(filtered_drift1)

    ax[1, 0].plot(filtered_drift0)
    ax[1, 0].set_title('Canal 1, fc = {}'.format(fc[1]))
    ax[1, 1].plot(filtered_drift1)
    ax[1, 1].set_title('Canal 1, fc = {}'.format(fc[1]))

    if show:
        plt.show()

    plt.close()

    # # add a 'best fit' line
    # plt.figure()
    # y = norm.pdf(np.linspace(min(filtered_noise0), max(filtered_noise0)), mu0, sigma0)
    # plt.hist(filtered_noise0, 100, range=(-0.005, 0.005), density=True)
    # plt.plot(np.linspace(-0.005, 0.005), y, 'r--', linewidth=2)
    # plt.show()
