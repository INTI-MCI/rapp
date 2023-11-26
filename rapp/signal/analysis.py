import os
import re
import decimal
import logging

import numpy as np
import pandas as pd

# from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

from scipy import signal
from scipy.stats import norm
from scipy.optimize import curve_fit

from rapp.signal.plot import Plot
from rapp.signal.phase import phase_difference
from rapp.utils import create_folder, round_to_n
from rapp import constants as ct

logger = logging.getLogger(__name__)


CHANNELS = [0, 1]


def poly_2(x, A, B, C):
    return A * x**2 + B * x + C


def detrend_poly(data, func):
    popt, pcov = curve_fit(func, np.arange(data.size), data)
    return data - func(np.arange(data.size), *popt)


def plot_histogram_and_pdf(data,  bins='quantized', prefix='', show=False):
    # PLOT THE HISTOGRAM AND PDF
    f, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for i, ax in enumerate(axs):
        channel_data = data[:, i]

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        # Plot the PDF.
        mu, sigma = norm.fit(channel_data)

        mu_rounded = round_to_n(mu, 1)
        sigma_rounded = round_to_n(sigma, 1)

        logger.info(
            "{} noise (mu, sigma) = ({}, {})"
            .format(i, mu_rounded, sigma_rounded))

        pdf_x = np.linspace(min(channel_data), max(channel_data), 100)
        pdf_y = norm.pdf(pdf_x, mu, sigma)

        channel_bins = bins
        if channel_bins == 'quantized':
            # We create the list of bins, knowing we have dicretization.
            d = np.diff(np.unique(channel_data)).min()
            left_of_first_bin = channel_data.min() - float(d) / 2
            right_of_last_bin = channel_data.max() + float(d) / 2
            channel_bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

            logger.info("Discretization step: {}".format(d))

        counts, edges = np.histogram(channel_data, bins=channel_bins, density=False)
        # Normalize histogram according to PDF.
        counts = (counts / np.max(counts)) * np.max(pdf_y)

        ax.bar(edges[:-1], counts, width=np.diff(edges), color='k', alpha=0.4, edgecolor='k')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        fit_label = "µ = {}.\nσ = {}.".format(mu_rounded, sigma_rounded)

        ax.set_xlabel(ct.LABEL_VOLTAGE)
        ax.set_title("Canal {}".format(i))

        ax.plot(pdf_x, pdf_y, 'k', linewidth=2, label=fit_label)
        ax.legend(loc='upper right', fontsize=10)

        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    f.savefig("{}-histogram".format(prefix))


def plot_noise_with_laser_off(output_folder, show=False):
    print("")
    logger.info("PROCESSING SIGNAL WITH LASER OFF (dark current)...")

    filename = 'dark-current.txt'

    SAMPLING_FREQUENCY = 59

    filepath = os.path.join(ct.INPUT_DIR, filename)

    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=(1, 2), encoding=ct.ENCONDIG)

    # PLOT THE RAW DATA
    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data[:, i]
        ax.set_ylabel(ct.LABEL_VOLTAGE)
        ax.set_xlabel(ct.LABEL_N_SAMPLE)
        ax.set_title("Canal {}".format(i))
        ax.plot(channel_data, '-', color='k')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    f.savefig("{}-signal".format(base_output_fname))

    if show:
        plt.show()

    # PLOT THE FFT
    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = data[:, i]
        fft = np.fft.fft(channel_data)

        xs = np.arange(0, len(fft))
        xs = (xs / len(fft)) * SAMPLING_FREQUENCY

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))
        ax.plot(xs[1:], fft[1:], color='k')

    f.tight_layout()
    f.savefig("{}-fft".format(base_output_fname))
    if show:
        plt.show()

    plt.close()

    plot_histogram_and_pdf(data, prefix=base_output_fname, show=show)

    if show:
        plt.show()

    plt.close()


def plot_noise_with_laser_on(output_folder, show=False):
    print("")
    logger.info("PROCESSING SIGNAL WITH LASER ON...")

    filename = 'laser-75-int-alta.txt'
    filepath = os.path.join(ct.INPUT_DIR, filename)
    data = np.loadtxt(filepath, delimiter=' ', skiprows=1, usecols=(1, 2), encoding=ct.ENCONDIG)

    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    f, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    # PLOT THE RAW DATA AND POLYNOMIAL FIT FOR THE DRIFT
    for i, ax in enumerate(axs):
        channel_data = data[:, i]
        xs = np.arange(channel_data.size)
        popt, pcov = curve_fit(poly_2, xs, channel_data)

        fitx = np.arange(min(xs), max(xs), step=0.01)
        fity = poly_2(fitx, *popt)

        ax.plot(channel_data, color='k')
        ax.plot(fitx, fity, '-', lw=2)

        if i == 0:
            ax.set_ylabel(ct.LABEL_VOLTAGE)

        ax.set_xlabel(ct.LABEL_N_SAMPLE)

        ax.set_title('Canal {}'.format(i))

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    f.savefig("{}-signal-and-fit".format(base_output_fname))

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, folder=output_folder)

    filtered = []
    for i in CHANNELS:
        data_detrend = detrend_poly(data[:, i], poly_2)

        b, a = signal.butter(3, 0.064, btype='highpass')
        filtered_noise = signal.filtfilt(b, a, data_detrend)

        filtered_noise[filtered_noise > 0.002] = 0
        filtered_noise[filtered_noise < -0.002] = 0

        filtered.append(filtered_noise)

        mu = np.mean(filtered_noise)
        std = np.std(filtered_noise)

        logger.info('µ ch0 = {}'.format(round_to_n(mu, 2)))
        logger.info('σ ch0 = {}'.format(round_to_n(std, 2)))

        plot.add_data(filtered_noise, style='-', label='Canal {}'.format(i))
        plot.legend()

    plot.save("{}-filtered-noise".format(filename[:-4]))

    filtered = np.array(filtered).T

    # quantization in histogram breaks in this case. Detects a step near zero and python hangs...
    # fix it!
    plot_histogram_and_pdf(filtered, bins=30, prefix=base_output_fname, show=show)

    if show:
        plt.show()

    plt.close()


def plot_drift(output_folder, show=False):
    print("")
    logger.info("PROCESSING LASER DRIFT...")

    reencendido = np.loadtxt(
        "data/laser-16-reencendido-1M.txt",
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


def plot_phase_difference(filepath, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    # TODO: maybe we can write these parameters in the header of the file,
    # TODO: so we don't have to parse them form the filename...
    cycles, step, samples = re. findall(r'\d+(?:\.\d+)?', filepath)

    cols = (0, 1, 2)
    data = pd.read_csv(filepath, delimiter=' ', header=0, usecols=cols, encoding=ct.ENCONDIG)
    data = data.groupby(['ANGLE'], as_index=False).agg({
        'A0': ['mean', 'std'],
        'A1': ['mean', 'std']
    })

    if len(data.index) == 1:
        raise ValueError("This is a file with only one angle!.")

    xs = np.deg2rad(np.array(data['ANGLE']))
    s1 = np.array(data['A0']['mean'])
    s2 = np.array(data['A1']['mean'])

    s1_sigma = np.array(data['A0']['std'])
    s2_sigma = np.array(data['A1']['std'])

    s1err = s1_sigma / np.sqrt(int(samples))
    s2err = s2_sigma / np.sqrt(int(samples))

    res = phase_difference(xs * 2, s1, s2, s1_sigma=s1_sigma, s2_sigma=s2_sigma, method='fit')

    error_deg = np.rad2deg(res.error)
    error_deg_rounded = round_to_n(error_deg, 2)

    # Obtain number of decimal places of the error:
    d = abs(decimal.Decimal(str(error_deg_rounded)).as_tuple().exponent)

    phase_diff_deg = np.rad2deg(res.value)
    phase_diff_deg_rounded = round(phase_diff_deg, d)

    title = "cycles={}, step={}, samples={}.".format(cycles, step, samples)
    phi_label = "φ=({} ± {})°.".format(phase_diff_deg_rounded, error_deg_rounded)

    logger.info(
        "Detected phase difference: {}"
        .format(phi_label)
    )

    logger.info(title)

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, title=title, folder=output_folder)

    plot.add_data(xs, s1, yerr=s1err, ms=6, color='k', mew=0.5, xrad=True, markevery=5, alpha=0.8)
    plot.add_data(xs, s2, yerr=s2err, ms=6, color='k', mew=0.5, xrad=True, markevery=5, alpha=0.8)

    fitx = res.fitx / 2

    if res.fitx is not None:
        plot.add_data(fitx, res.fits1, style='-', color='k', lw=1.5, xrad=True, label=phi_label)
        plot.add_data(fitx, res.fits2, style='-', color='k', lw=1.5, xrad=True)

    # plot._ax.set_xlim(0, 1)
    plot.legend(loc='upper right')
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    plot.save(filename="{}.png".format(os.path.basename(filepath)[:-4]))

    if show:
        plot.show()

    plot.close()

    return phase_diff_deg


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
        '2-full-cycles.txt',
        '2-full-cycles-2.txt',
        '1-full-cycles.txt',
        'test-clear-buffer.txt',
        'test-clear-buffer2.txt',
        'test-cycles2-step1.0-samples50.txt'
    ]

    for filename in filenames:
        filepath = os.path.join(ct.INPUT_DIR, filename)

        logger.info("Filepath: {}".format(filepath))
        plot_two_signals(filepath, output_folder, delimiter=' ', show=show)


def plot_two_signals(filepath, output_folder, delimiter='\t', usecols=(0, 1, 2), show=False):
    data = pd.read_csv(
        filepath, delimiter=delimiter, header=0, usecols=usecols, encoding=ct.ENCONDIG
    )

    data = data.groupby(['ANGLE']).mean().reset_index()

    xs = np.deg2rad(np.array(data['ANGLE']))
    s1 = np.array(data['A0'])
    s2 = np.array(data['A1'])

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=output_folder)
    plot.add_data(xs, s1, color='k', style='o-', alpha=1, mew=1)
    plot.add_data(xs, s2, color='k', style='o-', alpha=1, mew=1)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    plot.save(filename="{}.png".format(os.path.basename(filepath)[:-4]))

    if show:
        plot.show()

    plot.close()


def main(show):
    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    plot_noise_with_laser_off(output_folder, show=show)
    plot_noise_with_laser_on(output_folder, show=show)
    # plot_drift(output_folder, show=show)
    # plot_signals_per_n_measurement(output_folder, show=show)
    # plot_signals_per_angle(output_folder, show=show)


if __name__ == '__main__':
    main()
