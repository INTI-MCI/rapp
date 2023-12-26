import os
import re
import decimal
import logging

import numpy as np
import pandas as pd

from uncertainties import ufloat
from matplotlib import pyplot as plt

from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit

from rapp.signal.plot import Plot
from rapp.signal.phase import phase_difference
from rapp.utils import create_folder, round_to_n
from rapp import constants as ct

logger = logging.getLogger(__name__)


CHANNELS = [0, 1]
COLUMN_CH0 = 'CH0'
COLUMN_CH1 = 'CH1'
COLUMN_ANGLE = 'ANGLE'

REGEX_NUMBER_AFTER_WORD = r"(?<={word})\d+(?:\.\d+)?"

PARAMETER_STRING = "cycles={}, step={}, samples={}."

COVERAGE_FACTOR = 3

ANALYSIS_NAMES = [
    'all',
    'darkcurrent',
    'noise',
    'drift',
]


FILE_PARAMS = {
    "darkcurrent-range4V-samples40000-sps59.txt": {
        "sps": 59.5,
        "sep": " ",
        "band_stop_freq": [(9.45, 0.1), (9.45 * 3, 0.1), (6, 0.1)],
        "high_pass_freq": None,
        "outliers": None,
        "bins": "quantized"
    },
    "darkcurrent-range2V-samples100000.txt": {
        "sps": 838,
        "sep": "\t",
        "band_stop_freq": [(50, 10), (100, 1), (150, 1)],
        "high_pass_freq": None,
        "outliers": None,
        "bins": "quantized"
    },
    "darkcurrent-range4V-samples100000.txt": {
        "sps": 835,
        "sep": r"\s+",
        "band_stop_freq": [
            (50 * x, 0.5) for x in range(1, 6)] + [
            (131.55, 0.5)] + [
            (1 * x, 0.1) for x in range(1, 7)] + [
            (1 * x, 0.1) for x in range(8, 14)] + [
            (1 * x, 0.1) for x in range(15, 20)] + [
            (23, 0.1), (24, 0.1)],
        "high_pass_freq": 2,
        "outliers": None,
        "bins": 35
        # "bins": "quantized"
    },
    "continuous-range4V-584nm-samples10000-sps59.txt": {
        "sps": 59.5,
        "sep": " ",
        "band_stop_freq": None,  # [(9.4, 0.3), (18.09, 0.3), (18.8, 0.3), (28.22, 0.3)],
        "high_pass_freq": 2,
        "outliers": None,
        "bins": "quantized"
    },
    "continuous-range4V-632nm-samples100000.txt": {
        "sps": 847,  # This fits OK with line frequencies.
        "sep": r"\s+",
        "band_stop_freq": [(50 * x, 10 / x) for x in range(1, 8)],
        "high_pass_freq": 2,
        "outliers": [-0.0021, 0.0021],
        "bins": "quantized"
    }
}


def linear(f, a, b):
    return - a * f + b


def pink_noise(f, alpha, a):
    return a / f ** alpha


def read_measurement_file(filepath, sep=r"\s+"):
    return pd.read_csv(
        filepath,
        sep=sep, skip_blank_lines=True, comment='#', usecols=(0, 1, 2), encoding=ct.ENCONDIG
    )


def poly_2(x, A, B, C):
    return A * x**2 + B * x + C


def detrend_poly(data, func):
    popt, pcov = curve_fit(func, np.arange(data.size), data)
    return data - func(np.arange(data.size), *popt)


def parse_input_parameters_from_filepath(filepath):
    return (
        re.findall(REGEX_NUMBER_AFTER_WORD.format(word="cycles"), filepath)[0],
        re.findall(REGEX_NUMBER_AFTER_WORD.format(word="step"), filepath)[0],
        re.findall(REGEX_NUMBER_AFTER_WORD.format(word="samples"), filepath)[0]
    )


def generate_bins(data, step=0.125e-3):
    centers = np.arange(data.min(), data.max() + step, step)
    edges = centers - step / 2
    edges = np.append(edges, max(edges) + step)

    return centers, edges


def average_data(data):
    data = data.groupby([COLUMN_ANGLE], as_index=False)

    group_size = int(data.size()['size'][0])

    data = data.agg({
        COLUMN_CH0: ['mean', 'std'],
        COLUMN_CH1: ['mean', 'std']
    })

    xs = np.array(data[COLUMN_ANGLE])
    s1 = np.array(data[COLUMN_CH0]['mean'])
    s2 = np.array(data[COLUMN_CH1]['mean'])

    s1err = np.array(data[COLUMN_CH0]['std']) / np.sqrt(int(group_size))
    s2err = np.array(data[COLUMN_CH1]['std']) / np.sqrt(int(group_size))

    return xs, s1, s2, s1err, s2err


def plot_histogram_and_pdf(data,  bins='quantized', prefix='', show=False):
    """Plots a histogram and PDF for 2-channel data."""
    f, axs = plt.subplots(
        1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True, sharex=True)

    for i, ax in enumerate(axs):
        channel_data = data[:, i]

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        # Gaussian PDF.
        mu, sigma = stats.norm.fit(channel_data)

        mu_rounded = round_to_n(mu, 1)
        sigma_rounded = round_to_n(sigma, 1)

        logger.info(
            "CH{} noise (mu, sigma) = ({}, {})"
            .format(i, mu_rounded, sigma_rounded))

        pdf_x = np.linspace(min(channel_data), max(channel_data), num=1000)
        pdf_y = stats.norm.pdf(pdf_x, mu, sigma)

        channel_bins = bins[i] if isinstance(bins, list) else bins
        if channel_bins == 'quantized':
            centers, edges = generate_bins(channel_data, step=0.125e-3)
            counts, edges = np.histogram(channel_data, bins=edges, density=True)
        else:
            counts, edges = np.histogram(channel_data, bins=channel_bins, density=True)
            centers = (edges + np.diff(edges)[0] / 2)[:-1]

        logger.info("CH{} - Amount of bins: {}".format(i, len(centers)))

        ax.bar(centers, counts, width=np.diff(edges), color='k', alpha=0.2, edgecolor='k')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        fit_label = "µ = {:.1E}.\nσ = {}.".format(mu_rounded, sigma_rounded)

        ax.set_xlabel(ct.LABEL_VOLTAGE)
        ax.set_title("Canal {}".format(i))

        ax.plot(pdf_x, pdf_y, 'k', lw=2, label=fit_label)
        ax.legend(loc='upper right', fontsize=10)

        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    f.tight_layout()
    f.subplots_adjust(wspace=0.03)
    f.savefig("{}-histogram.png".format(prefix))


def plot_noise_with_laser_off(output_folder, show=False):
    print("")
    logger.info("PROCESSING SIGNAL WITH LASER OFF (dark current)...")

    # filename = "darkcurrent-range4V-samples40000-sps59.txt"
    # filename = "darkcurrent-range2V-samples100000.txt"
    filename = "darkcurrent-range4V-samples100000.txt"

    sps, sep, bstop, hpass, outliers, bins = FILE_PARAMS[filename].values()
    filepath = os.path.join(ct.INPUT_DIR, filename)

    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    data = read_measurement_file(filepath, sep=sep)

    logger.info("Plotting raw data...")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True)

    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]
        channel_data = channel_data - np.mean(channel_data)  # center signal
        res = stats.normaltest(channel_data)
        logger.info("Gaussian Test. p-value: {}".format(res.pvalue))

        ax.set_ylabel(ct.LABEL_VOLTAGE)
        ax.set_xlabel(ct.LABEL_N_SAMPLE)
        ax.set_title("Canal {}".format(i))
        ax.plot(channel_data, '-', color='k')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    f.subplots_adjust(hspace=0)
    f.tight_layout()

    f.savefig("{}-signal.png".format(base_output_fname))

    # if show:
    #    plt.show()

    plt.close()

    logger.info("Plotting FFT...")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

        psd = (2 * np.abs(np.fft.fft(channel_data)) ** 2) / (sps * len(channel_data))

        xs = np.arange(0, len(psd))
        xs = (xs / len(psd)) * sps
        N = len(xs)

        end = N // 2
        end = np.where(xs < 10)[0][-1]  # 1/f noise only below 10 Hz

        pink_x = xs[1:end]

        popt, pcov = curve_fit(linear, np.log(pink_x), np.log(psd[1:end]))
        # popt, pcov = curve_fit(pink_noise, pink_x, psd[1:end])

        us = np.sqrt(np.diag(pcov))
        alpha = popt[0]

        alpha_u = round_to_n(us[0], 1)
        # Obtain number of decimal places of the u:
        d = abs(decimal.Decimal(str(alpha_u)).as_tuple().exponent)
        alpha = round(alpha, d)

        logger.info("A/f^α noise estimation: α = {} ± {}".format(alpha, 0))
        logger.info("A/f^α noise estimation: scale = {} ± {}".format(popt[1], us[1]))

        label = "Ajuste 1/fᵅ (α = {:.2f} ± {:.2f})".format(alpha, alpha_u)

        if i == 0:
            ax.set_ylabel(ct.LABEL_PSD)

        pink_y = pink_noise(pink_x, alpha, np.exp(popt[1]))

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))
        ax.loglog(xs[:N // 2], psd[:N // 2], color='k')
        ax.loglog(pink_x, pink_y, color='deeppink', lw=2, label=label)

        line_frequencies = [50 * x for x in range(1, 6)]
        for i, freq in enumerate(line_frequencies, 0):
            freq_label = None
            if i == 0:
                freq_label = "50 Hz y armónicos"
            ax.axvline(x=freq, ls='--', lw=1, label=freq_label)

        line_frequencies = list(range(1, 7)) + list(range(8, 14)) + list(range(15, 20))
        for i, freq in enumerate(line_frequencies, 0):
            freq_label = None
            if i == 0:
                freq_label = "1 Hz y armónicos"
            ax.axvline(x=freq, ls='--', lw=1, color='C01', alpha=0.5, label=freq_label)

        ax.legend(loc='upper right', fontsize=11)

    f.subplots_adjust(wspace=0.03)
    f.tight_layout()
    f.savefig("{}-fft.png".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    filtered = []
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

        if i == 0:
            ax.set_ylabel(ct.LABEL_VOLTAGE)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))

        if bstop is not None:
            logger.info("Applying band-stop filtering on line frequencies...")
            for fr, delta in bstop:
                a = signal.butter(
                    2, [fr - delta, fr + delta], btype='bandstop', fs=sps, output='sos')
                channel_data = signal.sosfilt(a, channel_data)

        if hpass is not None:
            logger.info("Applying high-pass filtering on {} Hz...".format(hpass))
            b, a = signal.butter(3, hpass, btype='highpass', fs=sps)
            channel_data = signal.filtfilt(b, a, channel_data)

        if outliers is not None:
            channel_data[channel_data < outliers[0]] = 0
            channel_data[channel_data > outliers[1]] = 0

        fft = np.fft.fft(channel_data)

        xs = np.arange(0, len(fft))
        xs = (xs / len(fft)) * sps
        N = len(xs)
        ax.semilogy(xs[:N // 2], np.abs(fft[:N // 2]), color='k')
        ax.set_xlim(0, xs[-1] // 2)

        res = stats.normaltest(channel_data)
        logger.info("Gaussian Test. p-value: {}".format(res.pvalue))

        channel_data = channel_data - np.mean(channel_data)  # center signal
        filtered.append(channel_data)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    f.subplots_adjust(wspace=0.03)
    f.tight_layout()
    f.savefig("{}-filtered-fft.png".format(base_output_fname))

    # if show:
    #    plt.show()

    plt.close()

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, ysci=True, folder=output_folder)

    for i, filtered_channel in enumerate(filtered, 0):
        plot.add_data(filtered_channel, style='-', label='Canal {}'.format(i))

    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    height = max(abs(data.max()))
    plot._ax.set_ylim(-height / 2, height / 2)
    plot.legend(loc='upper right')

    plot.save("{}-filtered-signal.png".format(filename[:-4]))

    # if show:
    #    plt.show()

    plt.close()

    data = np.array(filtered).T
    plot_histogram_and_pdf(data, bins=bins, prefix=base_output_fname, show=show)

    if show:
        plt.show()

    plt.close()


def plot_noise_with_laser_on(output_folder, show=False):
    print("")
    logger.info("ANALYZING NOISE WITH LASER ON...")

    # filename = "continuous-range4V-584nm-samples10000-sps59.txt"
    filename = "continuous-range4V-632nm-samples100000.txt"

    filepath = os.path.join(ct.INPUT_DIR, filename)

    sps, sep, bstop, hpass, outliers, bins = FILE_PARAMS[filename].values()

    data = read_measurement_file(filepath, sep=sep)
    data = data[1:]
    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    logger.info("Fitting raw data to 2-degree polynomial...")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

        d = np.diff(np.unique(channel_data)).min()
        logger.info("Discretization step: {}".format(d))

        xs = np.arange(channel_data.size)
        popt, pcov = curve_fit(poly_2, xs, channel_data)

        fitx = np.arange(min(xs), max(xs), step=0.01)
        fity = poly_2(fitx, *popt)

        me = 100
        slice_data = channel_data[0::me]

        ax.plot(
            channel_data, 'o', color='k', mfc='None', ms=4, markevery=me, alpha=0.6, label="Datos")

        ax.plot(fitx, fity, '-', lw=2, label="Ajuste polinomial")
        ax.set_ylim(min(slice_data), max(slice_data))
        ax.legend(loc='lower left')

        if i == 0:
            ax.set_ylabel(ct.LABEL_VOLTAGE)

        ax.set_xlabel(ct.LABEL_N_SAMPLE)

        ax.set_title('Canal {}'.format(i))

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    f.subplots_adjust(hspace=0)
    f.tight_layout()
    f.savefig("{}-poly-fit".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    logger.info("Removing 2-degree polynomial from raw data...")
    for i in CHANNELS:
        data['CH{}'.format(i)] = detrend_poly(data['CH{}'.format(i)], poly_2)

    logger.info("Plotting FFT...")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

        psd = (2 * np.abs(np.fft.fft(channel_data)) ** 2) / (sps * len(channel_data))

        xs = np.arange(0, len(psd))
        xs = (xs / len(psd)) * sps
        N = len(xs)

        end = N // 2
        end = np.where(xs < 10)[0][-1]  # 1/f noise only below 10 Hz

        pink_x = xs[1:end]

        popt, pcov = curve_fit(linear, np.log(pink_x), np.log(psd[1:end]))
        # popt, pcov = curve_fit(pink_noise, pink_x, psd[1:end])

        us = np.sqrt(np.diag(pcov))

        alpha = popt[0]

        alpha_u = round_to_n(us[0], 1)
        # Obtain number of decimal places of the u:
        d = abs(decimal.Decimal(str(alpha_u)).as_tuple().exponent)
        alpha = round(alpha, d)

        logger.info("A/f^α noise estimation: α = {} ± {}".format(alpha, alpha_u))
        logger.info("A/f^α noise estimation: scale = {} ± {}".format(popt[1], us[1]))

        label = "Ajuste 1/fᵅ (α = {:.2f} ± {:.2f})".format(alpha, alpha_u)

        pink_y = pink_noise(xs[1:end], alpha, np.exp(popt[1]))[:end]

        if i == 0:
            ax.set_ylabel(ct.LABEL_PSD)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))
        ax.loglog(xs[1:N // 2], psd[1:N // 2], color='k')
        ax.loglog(pink_x, pink_y, color='deeppink', lw=2, label=label)

        line_frequencies = [50 * x for x in range(1, int(xs[N // 2] / 50))]
        for i, freq in enumerate(line_frequencies, 0):
            freq_label = None
            if i == 0:
                freq_label = "50 Hz y armónicos"
            ax.axvline(x=freq, ls='--', lw=1, label=freq_label)

        ax.legend(loc='upper right', fontsize=10)

    for ax in axs.flat:
        ax.label_outer()

    f.subplots_adjust(hspace=0)
    f.tight_layout()
    f.savefig("{}-fft".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    filtered = []
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

        if i == 0:
            ax.set_ylabel(ct.LABEL_VOLTAGE)

        ax.set_xlabel(ct.LABEL_FREQUENCY)

        if bstop is not None:
            logger.info("Filtering bandstop: {} Hz".format([i[0] for i in bstop]))
            for fr, delta in bstop:
                b, a = signal.butter(2, [fr - delta, fr + delta], btype='bandstop', fs=sps)
                channel_data = signal.lfilter(b, a, channel_data)

        if hpass is not None:
            logger.info("Filtering highpass: {} Hz".format(hpass))

            b, a = signal.butter(3, hpass, btype='highpass', fs=sps)
            channel_data = signal.filtfilt(b, a, channel_data)

        if outliers is not None:
            channel_data[channel_data < outliers[0]] = 0
            channel_data[channel_data > outliers[1]] = 0

        res = stats.normaltest(channel_data)
        logger.info("Gaussian Test. p-value: {}".format(res.pvalue))

        filtered.append(channel_data)

        psd = np.abs(np.fft.fft(channel_data)) ** 2

        pink_x = xs[1:end]

        popt, pcov = curve_fit(linear, np.log(pink_x), np.log(psd[1:end]))
        # popt, pcov = curve_fit(pink_noise, pink_x, psd[1:end])

        us = np.sqrt(np.diag(pcov))

        alpha = popt[0]

        alpha_u = round_to_n(us[0], 1)
        # Obtain number of decimal places of the u:
        d = abs(decimal.Decimal(str(alpha_u)).as_tuple().exponent)
        alpha = round(alpha, d)

        logger.info("A/f^α noise estimation: α = {} ± {}".format(alpha, alpha_u))
        logger.info("A/f^α noise estimation: scale = {} ± {}".format(popt[1], us[1]))

        label = "Ajuste 1/fᵅ (α = {} ± {})".format(alpha, alpha_u)

        pink_y = pink_noise(xs[1:end], alpha, np.exp(popt[1]))[:end]

        xs = np.arange(0, len(psd))
        xs = (xs / len(psd)) * sps
        N = len(xs)
        ax.loglog(xs[:N // 2], psd[:N // 2], color='k')
        ax.loglog(pink_x, pink_y, color='deeppink', lw=2, label=label)

    f.subplots_adjust(hspace=0)
    f.tight_layout()
    f.savefig("{}-filtered-fft".format(base_output_fname))

    # if show:
    #    plt.show()

    plt.close()

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, ysci=True, folder=output_folder)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    for i, filtered_channel in enumerate(filtered, 0):
        plot.add_data(filtered_channel, style='-', label='Canal {}'.format(i))

    height = max(abs(data.max()))
    plot._ax.set_ylim(-height / 20, height / 20)
    plot.legend(loc='upper right')

    plot.save("{}-filtered-signal.png".format(filename[:-4]))

    plt.close()

    data = np.array(filtered).T

    plot_histogram_and_pdf(data, bins=bins, prefix=base_output_fname, show=show)

    if show:
        plt.show()

    plt.close()


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


def averaged_phase_difference(folder, method):
    logger.info("Calculating phase difference for {}...".format(folder))

    files = [os.path.join(folder, x) for x in os.listdir(folder)]

    values = []
    for filepath in files:
        cycles, step, samples = parse_input_parameters_from_filepath(filepath)

        data = read_measurement_file(filepath)

        if len(data.index) == 1:
            raise ValueError("This is a file with only one angle!.")

        xs, s1, s2, s1err, s2err = average_data(data)

        x_sigma = ct.ANALYZER_MIN_STEP / (2 * np.sqrt(3))

        res = phase_difference(
            np.deg2rad(xs) * 2, s1, s2, x_sigma=x_sigma, s1_sigma=s1err, s2_sigma=s2err,
            method=method
        )

        phase_diff = res.value / 2
        phase_diff_u = res.u / 2

        phase_diff_u_rounded = round_to_n(phase_diff_u * COVERAGE_FACTOR, 2)

        # Obtain number of decimal places of the u:
        d = abs(decimal.Decimal(str(phase_diff_u_rounded)).as_tuple().exponent)

        phase_diff_rounded = round(phase_diff, d)

        phi_label = "φ=({} ± {})°.".format(phase_diff_rounded, phase_diff_u_rounded)
        title = "{}\ncycles={}, step={}, samples={}.".format(phi_label, cycles, step, samples)

        logger.info(
            "Detected phase difference: {}"
            .format(phi_label)
        )

        logger.info(title)

        values.append(ufloat(phase_diff, phase_diff_u))

    N = len(values)
    avg_phase_diff = sum(values) / N

    logger.info("Averaged phase difference over {} measurements: {}".format(N, avg_phase_diff))

    return avg_phase_diff


def optical_rotation(folder1, folder2, method):
    logger.info("Calculating optical rotation...")
    logger.info("Folder without optical active sample measurements {}...".format(folder1))
    logger.info("Folder with optical active sample measurements {}...".format(folder2))

    phase_diff_without_sample = averaged_phase_difference(folder1, method=method)
    phase_diff_with_sample = averaged_phase_difference(folder2, method=method)

    optical_rotation = (phase_diff_with_sample - phase_diff_without_sample) / 2

    logger.info("Optical rotation measured: {}".format(optical_rotation))

    return optical_rotation


def plot_phase_difference(filepath, method, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    cycles, step, samples = parse_input_parameters_from_filepath(filepath)

    data = read_measurement_file(filepath)

    if len(data.index) == 1:
        raise ValueError("This is a file with only one angle!.")

    xs, s1, s2, s1err, s2err = average_data(data)

    s1_sigma = None if np.isnan(s1err).any() else s1err
    s2_sigma = None if np.isnan(s1err).any() else s2err

    print(s1_sigma)
    print(s2_sigma)

    x_sigma = ct.ANALYZER_MIN_STEP / (2 * np.sqrt(3))

    res = phase_difference(
        np.deg2rad(xs) * 2, s1, s2, x_sigma=x_sigma, s1_sigma=s1err, s2_sigma=s2err, method=method
    )

    phase_diff = res.value / 2
    phase_diff_u = res.u / 2

    phase_diff_u_rounded = round_to_n(phase_diff_u * COVERAGE_FACTOR, 2)

    # Obtain number of decimal places of the u:
    d = abs(decimal.Decimal(str(phase_diff_u_rounded)).as_tuple().exponent)

    phase_diff_rounded = round(phase_diff, d)

    phi_label = "φ=({} ± {})°.".format(phase_diff_rounded, phase_diff_u_rounded)
    title = "{}\ncycles={}, step={}, samples={}.".format(phi_label, cycles, step, samples)

    logger.info(
        "Detected phase difference: {}"
        .format(phi_label)
    )

    logger.info(title)

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_DEGREE, title=title, folder=output_folder)

    plot.add_data(xs, s1, yerr=s1err, ms=6, color='k', mew=0.5, markevery=5, alpha=0.8)
    plot.add_data(xs, s2, yerr=s2err, ms=6, color='k', mew=0.5, markevery=5, alpha=0.8)

    if res.fitx is not None:
        fitx = res.fitx / 2
        plot.add_data(fitx, res.fits1, style='-', color='k', lw=1.5)
        plot.add_data(fitx, res.fits2, style='-', color='k', lw=1.5)

    # plt.legend(loc='upper right', fancybox=True)

    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    plot.save(filename="{}.png".format(os.path.basename(filepath)[:-4]))

    if show:
        plot.show()

    plot.close()

    return phase_diff


def plot_signals_per_n_measurement(output_folder, show=False):
    print("")
    logger.info("PLOTTING SIGNALS VS # OF MEASUREMENT.")

    filenames = [
        'laser-75-int-alta.txt',
        'laser-75-encendido-15min.txt',
        'laser-16-reencendido-1M.txt',
        'laser-16-75-grados-int-baja.txt'
    ]

    for filename in filenames:
        filepath = os.path.join(ct.INPUT_DIR, filename)
        logger.info("Filepath: {}".format(filepath))

        plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, folder=output_folder)
        plot.set_title(filename[:-4])

        cols = (0, 1, 2)
        data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=cols, encoding=ct.ENCONDIG)
        data = data[:, 1]
        xs = np.arange(1, data.size + 1, step=1)

        plot.add_data(xs, data, style='-', color='k')
        plot.save(filename=filename[:-4])

        if show:
            plot.show()

        plot.close()


def plot_signals_per_angle(output_folder, show=False):
    print("")
    logger.info("PLOTTING SIGNALS VS ANALYZER ANGLE...")

    filenames = [
        # '2-full-cycles.txt',
        # '2-full-cycles-2.txt',
        # '1-full-cycles.txt',
        # 'test-clear-buffer.txt',
        # 'test-clear-buffer2.txt',
        'test-cycles2-step1.0-samples50.txt'
    ]

    for filename in filenames:
        filepath = os.path.join(ct.INPUT_DIR, filename)

        logger.info("Filepath: {}".format(filepath))
        plot_two_signals(filepath, output_folder, sep=' ', show=show)


def plot_two_signals(filepath, output_folder, sep='\t', usecols=(0, 1, 2), show=False):
    data = read_measurement_file(filepath, sep=sep)
    data = data.groupby([COLUMN_ANGLE]).mean().reset_index()

    xs = np.deg2rad(np.array(data[COLUMN_ANGLE]))
    s1 = np.array(data[COLUMN_CH0])
    s2 = np.array(data[COLUMN_CH1])

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=output_folder)
    plot.set_title(PARAMETER_STRING.format(*parse_input_parameters_from_filepath(filepath)))
    plot.add_data(xs, s1, style='o-', alpha=1, mew=1, label='CH0')
    plot.add_data(xs, s2, style='o-', alpha=1, mew=1, label='CH1')
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plot.legend()

    plot.save(filename="{}.png".format(os.path.basename(filepath)[:-4]))

    if show:
        plot.show()

    plot.close()


def main(name, show):
    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # TODO: add another subparser and split these options in different commands with parameters
    if name not in ANALYSIS_NAMES:
        raise ValueError("Analysis with name {} not implemented".format(name))

    if name in ['all', 'darkcurrent']:
        plot_noise_with_laser_off(output_folder, show=show)

    if name in ['all', 'noise']:
        plot_noise_with_laser_on(output_folder, show=show)

    if name in ['all', 'drift']:
        plot_drift(output_folder, show=show)


if __name__ == '__main__':
    main()
