import os
import re
import logging

import numpy as np
import pandas as pd

# from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

from scipy import signal
from scipy.stats import norm
from scipy.optimize import curve_fit

from rapp.utils import create_folder, round_to_n
from rapp.signal.phase import phase_difference

from rapp.signal.plot import Plot

OUTPUT_FOLDER = 'output-plots'
INPUT_FOLDER = 'data'

LABEL_VOLTAGE = "Voltaje [V]"
LABEL_ANGLE = "Ángulo del rotador [rad]"
LABEL_N_SAMPLE = "No. de muestra"
LABEL_COUNTS = "Cuentas"


logger = logging.getLogger(__name__)


def pol1(x, A, B):
    return A * x + B


def pol2(x, A, B, C):
    return A * x**2 + B * x + C


def pol3(x, A, B, C, D):
    return A * x**3 + B * x**2 + C * x + D


def fit_and_plot(data, n_data, func, n, show=False):
    for i in np.arange(len(func)):
        popt, pcov = curve_fit(func[i], np.arange(data.size), data)
        plt.plot(np.arange(data.size), data)
        plt.plot(n_data, func[i](np.arange(data.size), *popt))
        plt.title('Ajuste deriva Canal {}'.format(n))
        if show:
            plt.show(block=True)
        logger.info(popt)
    return


def detrend_poly(data, func):
    popt, pcov = curve_fit(func, np.arange(data.size), data)
    # plt.plot(func(np.arange(data.size), *popt))
    # plt.plot(data)
    # plt.show()
    return data - func(np.arange(data.size), *popt)


def plot_signals_per_n_measurement(show=False):
    print("")
    logger.info("PLOTTING SIGNALS VS # OF MEASUREMENT.")

    filenames = [
        'laser-75-int-alta.txt',
        'laser-75-encendido-15min.txt',
        'laser-16-reencendido-1M.txt',
        'laser-16-75-grados-int-baja.txt'
    ]

    for filename in filenames:
        filepath = os.path.join('data', filename)
        logger.info("Filepath: {}".format(filepath))

        plot = Plot(ylabel=LABEL_VOLTAGE, xlabel="# measurement")
        plot.set_title(filename[:-4])

        cols = (0, 1, 2)
        data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding='iso-8859-1')
        data = data[:, 1]
        xs = np.arange(1, data.size + 1, step=1)

        plot.add_data(xs, data, style='-', color='k')
        plot.save(filename=filename[:-4])

        if show:
            plot.show()

        plot.close()


def plot_signals_per_angle(show=False):
    print("")
    logger.info("PLOTTING SIGNALS VS ANALYZER ANGLE...")

    filenames = [
        '2-full-cycles.txt',
        '2-full-cycles-2.txt',
        '1-full-cycles.txt',
        'test-clear-buffer.txt',
        'test-clear-buffer2.txt',
        'test-cycles2-step1.0-samples50.txt'
    ]

    for filename in filenames:
        filepath = os.path.join('data', filename)

        logger.info("Filepath: {}".format(filepath))

        cols = (0, 1, 2)
        data = np.loadtxt(filepath, skiprows=1, usecols=cols, encoding='iso-8859-1')
        voltage = data[:, 1]
        angles = np.deg2rad(data[:, 0])

        plot = Plot(ylabel=LABEL_VOLTAGE, xlabel=LABEL_ANGLE, title=filename[:-4])
        plot.add_data(angles, voltage, style='o-', color='k', xrad=True)
        plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

        plot.save(filename="{}.png".format(filename[:-4]))

        if show:
            plot.show()

        plot.close()


def plot_drift(show=False):
    print("")
    logger.info("PROCESSING LASER DRIFT...")

    file = np.loadtxt(
        'data/laser-75-int-alta.txt',
        delimiter=' ', skiprows=1, usecols=(1, 2), encoding='iso-8859-1')

    ch0 = file[:, 0]
    ch1 = file[:, 1]

    reencendido = np.loadtxt(
        "data/laser-16-reencendido-1M.txt",
        delimiter=' ', skiprows=1, usecols=(1, 2), encoding='iso-8859-1')

    r0 = reencendido[:, 0]
    r1 = reencendido[:, 1]

    data_detrend0 = detrend_poly(r0[200000:], pol1)
    data_detrend1 = detrend_poly(r1[200000:], pol2)

    plt.figure()
    fft_data0 = np.fft.fft(data_detrend0)
    fft_data1 = np.fft.fft(data_detrend1)
    plt.plot(data_detrend0)
    plt.plot(data_detrend1)

    if show:
        plt.show()

    plt.close()

    plt.figure()
    plt.semilogy(np.abs(fft_data0))
    plt.semilogy(np.abs(fft_data1))

    if show:
        plt.show()

    plt.close()

    fit_and_plot(ch0, np.arange(ch0.size), [pol1], 0, show=show)
    fit_and_plot(ch1, np.arange(ch1.size), [pol2], 1, show=show)

    data_detrend0 = detrend_poly(ch0, pol1)
    data_detrend1 = detrend_poly(ch1, pol2)

    b, a = signal.butter(3, 0.064, btype='highpass')
    filtered_noise0 = signal.filtfilt(b, a, data_detrend0)
    filtered_noise1 = signal.filtfilt(b, a, data_detrend1)

    mu_0 = np.mean(filtered_noise0)
    std_dev0 = np.std(filtered_noise0)
    std_dev1 = np.std(filtered_noise1)
    mu_1 = np.mean(filtered_noise1)

    logger.info('Media ch0 = {}'.format(round_to_n(mu_0, 2)))
    logger.info('Sigma ch0 = {}'.format(round_to_n(std_dev0, 2)))
    logger.info('Media ch1 = {}'.format(round_to_n(mu_1, 2)))
    logger.info('Sigma ch1 = {}'.format(round_to_n(std_dev1, 2)))

    plt.figure()
    plt.title('Ruido filtrado')
    plt.plot(filtered_noise0, label='Canal 0')
    plt.plot(filtered_noise1, label='Canal 1')
    plt.legend()

    if show:
        plt.show()

    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Ruido filtrado')
    ax[0].hist(filtered_noise0, 100, range=(-0.005, 0.005))
    ax[0].set_title('Canal 0')
    ax[1].hist(filtered_noise1, 100, range=(-0.003, 0.003))
    ax[1].set_title('Canal 1')

    if show:
        plt.show()

    plt.close()

    # # add a 'best fit' line
    # plt.figure()
    # y = norm.pdf(np.linspace(min(filtered_noise0), max(filtered_noise0)), mu0, sigma0)
    # plt.hist(filtered_noise0, 100, range=(-0.005, 0.005), density=True)
    # plt.plot(np.linspace(-0.005, 0.005), y, 'r--', linewidth=2)
    # plt.show()


def plot_dark_current(show=False):
    print("")
    logger.info("PROCESSING DARK CURRENT...")

    filename = 'dark-current.txt'
    filepath = os.path.join(INPUT_FOLDER, filename)

    base_output_fname = "{}".format(os.path.join(OUTPUT_FOLDER, filename[:-4]))

    cols = (0, 1, 2)
    data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding='iso-8859-1')

    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data[:, i + 1]
        ax.set_ylabel(LABEL_VOLTAGE)
        ax.set_xlabel(LABEL_N_SAMPLE)
        ax.set_title("Canal A{}".format(i))
        ax.plot(channel_data, '-', color='k')
        # ax.set_xlim(0, 500)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    f.savefig("{}-signal".format(base_output_fname))

    plt.close()

    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = data[:, i + 1]

        if i == 0:
            ax.set_ylabel(LABEL_COUNTS)

        ax.set_xlabel(LABEL_VOLTAGE)
        ax.set_title("Canal A{}".format(i))
        ax.hist(channel_data, color='k', alpha=0.4, edgecolor='k', density=True)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        # Plot the PDF.
        mu, sigma = norm.fit(channel_data)

        mu_rounded = round_to_n(mu, 1)
        sigma_rounded = round_to_n(sigma, 1)

        logger.info(
            "A{} noise (mu, sigma) = ({}, {})"
            .format(i, mu_rounded, sigma_rounded))

        xmin, xmax = ax.get_xlim()
        fit_xs = np.linspace(xmin, xmax, 100)
        fit_ys = norm.pdf(fit_xs, mu, sigma)
        fit_label = "µ = {}.\nσ = {}.".format(mu_rounded, sigma_rounded)

        ax.plot(fit_xs, fit_ys, 'k', linewidth=2, label=fit_label)
        ax.legend(loc='upper right', fontsize=10)

    f.savefig("{}-histogram".format(base_output_fname))

    if show:
        plt.show()

    plt.close()


def plot_phase_difference(filepath, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    # TODO: maybe we can write these parameters in the header of the file,
    # TODO: so we don't have to parse them form the filename...
    cycles, step, samples = re. findall(r'\d+(?:\.\d+)?', filepath)

    cols = (0, 1, 2)

    # data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding='iso-8859-1')
    data = pd.read_csv(filepath, delimiter=' ', header=0, usecols=cols, encoding='iso-8859-1')
    data = data.groupby(['ANGLE']).mean().reset_index()

    if len(data.index) == 1:
        raise ValueError("This is a file with only one angle!.")

    xs = np.deg2rad(np.array(data['ANGLE']))
    s1 = np.array(data['A0'])
    s2 = np.array(data['A1'])

    res = phase_difference(xs * 2, s1, s2, method='fit')
    phase_diff_rad_rounded = round_to_n(res.phase_diff, 3)

    phase_diff_deg = np.rad2deg(res.phase_diff)
    phase_diff_deg_rounded = round_to_n(phase_diff_deg, 3)

    title = "cycles={}, step={}, samples={}.".format(cycles, step, samples)
    label = "φ={} deg.".format(phase_diff_deg_rounded)

    logger.info(
        "Detected phase difference: {} deg. {} rad."
        .format(phase_diff_deg_rounded, phase_diff_rad_rounded)
    )

    logger.info(title)

    plot = Plot(ylabel=LABEL_VOLTAGE, xlabel=LABEL_ANGLE, title=title)

    plot.add_data(xs, s1, ms=6, color='k', mew=0.5, xrad=True, markevery=5, alpha=0.8)
    plot.add_data(xs, s2, ms=6, color='k', mew=0.5, xrad=True, markevery=5, alpha=0.8)

    fitx = res.fitx / 2

    if res.fitx is not None:
        plot.add_data(fitx, res.fits1, style='-', color='k', lw=1.5, xrad=True, label=label)
        plot.add_data(fitx, res.fits2, style='-', color='k', lw=1.5, xrad=True)

    # plot._ax.set_xlim(0, 1)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plot.legend(loc='upper right')
    basename = os.path.basename(filepath)
    plot.save(filename="{}.png".format(basename[:-4]))

    if show:
        plot.show()

    plot.close()

    return phase_diff_deg


def main(show):
    create_folder(OUTPUT_FOLDER)

    plot_dark_current(show=show)
    plot_drift(show=show)
    plot_signals_per_n_measurement(show=show)
    plot_signals_per_angle(show=show)


if __name__ == '__main__':
    main()
