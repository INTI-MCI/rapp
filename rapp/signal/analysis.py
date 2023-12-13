import os
import re
import decimal
import logging

import numpy as np
import pandas as pd

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


def plot_histogram_and_pdf(data,  bins='quantized', prefix='', show=False):
    """Plots a histogram and PDF for 2-channel data."""
    f, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True, sharex=True)

    for i, ax in enumerate(axs):
        channel_data = data[:, i]

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        # Plot the PDF.
        mu, sigma = stats.norm.fit(channel_data)

        mu_rounded = round_to_n(mu, 1)
        sigma_rounded = round_to_n(sigma, 1)

        logger.info(
            "CH{} noise (mu, sigma) = ({}, {})"
            .format(i, mu_rounded, sigma_rounded))

        pdf_x = np.arange(min(channel_data), max(channel_data), 0.00001)
        pdf_y = stats.norm.pdf(pdf_x, mu, sigma)

        channel_bins = bins
        if channel_bins == 'quantized':
            # We create the list of bins, knowing we have dicretization.
            d = np.diff(np.unique(channel_data)).min()
            left_of_first_bin = channel_data.min() - float(d) / 2
            right_of_last_bin = channel_data.max() + float(d) / 2
            channel_bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

            logger.info("Discretization step: {}".format(d))

        counts, edges = np.histogram(channel_data, bins=channel_bins, density=True)

        ax.bar(edges[:-1], counts, width=np.diff(edges), color='k', alpha=0.2, edgecolor='k')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        fit_label = "µ = {:.1E}.\nσ = {}.".format(mu_rounded, sigma_rounded)

        ax.set_xlabel(ct.LABEL_VOLTAGE)
        ax.set_title("Canal {}".format(i))

        ax.plot(pdf_x, pdf_y, 'k', lw=2, label=fit_label)
        ax.legend(loc='upper right', fontsize=10)

        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    f.savefig("{}-histogram.png".format(prefix))


def plot_noise_with_laser_off(output_folder, show=False):
    print("")
    logger.info("PROCESSING SIGNAL WITH LASER OFF (dark current)...")

    file_params = 'darkcurrent-range4V-samples40000.txt', 59.5, ' ', [(5.93, 0.15), (9.45, 0.15), (28.38, 0.15)], 0.5  # noqa
    # file_params = 'darkcurrent-range2V-samples100000.txt', 845, '\t', [(50, 10), (100, 1), (150, 1)], 26  # noqa

    filename, sps, sep, bpass, hpass = file_params
    filepath = os.path.join(ct.INPUT_DIR, filename)

    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    data = np.loadtxt(filepath, delimiter=sep, skiprows=1, usecols=(1, 2), encoding=ct.ENCONDIG)

    logger.info("Plotting raw data...")
    f, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data[:, i]
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

    f.savefig("{}-signal.png".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    logger.info("Plotting FFT...")
    f, axs = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data[:, i]

        fft = np.fft.fft(channel_data)

        xs = np.arange(0, len(fft))
        xs = (xs / len(fft)) * sps

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))
        ax.semilogy(xs, np.abs(fft), color='k')

    f.tight_layout()

    f.savefig("{}-fft.png".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    filtered = []

    logger.info("Filtering unwanted frequencies...")
    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data[:, i]

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))

        for fr, delta in bpass:
            b, a = signal.butter(3, [fr - delta, fr + delta], btype='bandstop', fs=sps)
            channel_data = signal.lfilter(b, a, channel_data)

        b, a = signal.butter(3, hpass, btype='highpass', fs=sps)
        channel_data = signal.filtfilt(b, a, channel_data)

        fft = np.fft.fft(channel_data)

        xs = np.arange(0, len(fft))
        xs = (xs / len(fft)) * sps
        ax.semilogy(xs, abs(fft), color='k')

        res = stats.normaltest(channel_data)
        logger.info("Gaussian Test. p-value: {}".format(res.pvalue))

        filtered.append(channel_data)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    f.tight_layout()
    f.savefig("{}-filtered-fft.png".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i, filtered_channel in enumerate(filtered, 0):
        axs[i].plot(filtered_channel, color='k')
        axs[i].set_ylabel(ct.LABEL_VOLTAGE)
        axs[i].set_xlabel(ct.LABEL_N_SAMPLE)
        axs[i].ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(3))

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    f.tight_layout()
    f.savefig("{}-filtered-signal.png".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    data = np.array(filtered).T
    plot_histogram_and_pdf(data, bins='auto', prefix=base_output_fname, show=show)

    if show:
        plt.show()

    plt.close()


def plot_noise_with_laser_on(output_folder, show=False):
    print("")
    logger.info("ANALYZING NOISE WITH LASER ON...")

    filename, sep, sps, bpass, hpass = 'continuous-584nm-int-alta-samples10000.txt', ' ', 59.5, [(9.4, 0.3), (18.09, 0.3), (18.8, 0.3), (28.22, 0.3)], 2  # noqa
    # filename, sep, sps, bpass, hpass = (
    #    '2023-12-07-continuous-632nm-cycles0-step10-samples100000.txt', r"\s+", 845, [(50, 20), (100, 10), (150, 10), (200, 10), (250, 10), (300, 10), (350, 10), (400, 10)], 25  # noqa
    # )

    filepath = os.path.join(ct.INPUT_DIR, filename)

    data = read_measurement_file(filepath, sep=sep)
    data = data[1:]
    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    logger.info("Fitting raw data to 2-degree polynomial...")
    f, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

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

    f.savefig("{}-poly-fit".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    logger.info("Removing 2-degree polynomial from signals...")
    for i in CHANNELS:
        data['CH{}'.format(i)] = detrend_poly(data['CH{}'.format(i)], poly_2)

    logger.info("Plotting FFT...")
    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]
        channel_fft = np.fft.fft(channel_data)

        xs = np.arange(0, len(channel_fft))
        xs = (xs / len(channel_fft)) * sps

        if i == 0:
            ax.set_ylabel(ct.LABEL_COUNTS)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))
        ax.semilogy(xs, np.abs(channel_fft), color='k')

    f.savefig("{}-fft".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    filtered = []
    logger.info("Filtering unwanted frequencies...")
    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = data['CH{}'.format(i)]

        for fr, delta in bpass:
            b, a = signal.butter(2, [fr - delta, fr + delta], btype='bandstop', fs=sps)
            channel_data = signal.lfilter(b, a, channel_data)

        b, a = signal.butter(3, hpass, btype='highpass', fs=sps)
        channel_data = signal.filtfilt(b, a, channel_data)

        channel_data[channel_data > 0.00125] = 0
        channel_data[channel_data < -0.00125] = 0

        data['CH{}'.format(i)] = channel_data

        fft = np.fft.fft(channel_data)

        xs = np.arange(0, len(fft))
        xs = (xs / len(fft)) * sps
        ax.semilogy(xs, abs(fft), color='k')

        res = stats.normaltest(channel_data)
        logger.info("Gaussian Test. p-value: {}".format(res.pvalue))

        filtered.append(channel_data)

    f.savefig("{}-filtered-fft".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, ysci=True, folder=output_folder)
    for i, filtered_channel in enumerate(filtered, 0):

        plot.add_data(filtered_channel, style='-', label='Canal {}'.format(i))
        plot.legend()
        plot._ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    f.tight_layout()
    f.savefig("{}-filtered-signal.png".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    data = data[['CH0', 'CH1']].to_numpy()

    plot_histogram_and_pdf(data, bins='auto', prefix=base_output_fname, show=show)

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


def plot_phase_difference(filepath, method, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    cycles, step, samples = parse_input_parameters_from_filepath(filepath)

    data = pd.read_csv(
        filepath,
        sep=r"\s+", skip_blank_lines=True, comment='#', header=0, usecols=(0, 1, 2),
        encoding=ct.ENCONDIG
    )

    data = data.groupby([COLUMN_ANGLE], as_index=False).agg({
        COLUMN_CH0: ['mean', 'std'],
        COLUMN_CH1: ['mean', 'std']
    })

    if len(data.index) == 1:
        raise ValueError("This is a file with only one angle!.")

    xs = np.deg2rad(np.array(data[COLUMN_ANGLE]))
    s1 = np.array(data[COLUMN_CH0]['mean'])
    s2 = np.array(data[COLUMN_CH1]['mean'])

    s1err = np.array(data[COLUMN_CH0]['std']) / np.sqrt(int(samples))
    s2err = np.array(data[COLUMN_CH1]['std']) / np.sqrt(int(samples))

    x_sigma = ct.ANALYZER_MIN_STEP / (2 * np.sqrt(3))

    res = phase_difference(
        xs * 2, s1, s2, x_sigma=x_sigma, s1_sigma=s1err, s2_sigma=s2err, method=method)

    error_deg = np.rad2deg((res.u / 2) * COVERAGE_FACTOR)
    error_deg_rounded = round_to_n(error_deg, 2)

    # Obtain number of decimal places of the u:
    d = abs(decimal.Decimal(str(error_deg_rounded)).as_tuple().exponent)

    phase_diff_deg = np.rad2deg(res.value / 2)
    phase_diff_deg_rounded = round(phase_diff_deg, d)

    phi_label = "φ=({} ± {})°.".format(phase_diff_deg_rounded, error_deg_rounded)
    title = "{}\ncycles={}, step={}, samples={}.".format(phi_label, cycles, step, samples)

    logger.info(
        "Detected phase difference: {}"
        .format(phi_label)
    )

    logger.info(title)

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_DEGREE, title=title, folder=output_folder)

    xs = np.rad2deg(xs)
    plot.add_data(xs, s1, yerr=s1err, ms=6, color='k', mew=0.5, markevery=5, alpha=0.8)
    plot.add_data(xs, s2, yerr=s2err, ms=6, color='k', mew=0.5, markevery=5, alpha=0.8)

    if res.fitx is not None:
        fitx = res.fitx / 2
        fitx = np.rad2deg(fitx)
        plot.add_data(fitx, res.fits1, style='-', color='k', lw=1.5)
        plot.add_data(fitx, res.fits2, style='-', color='k', lw=1.5)

    # plt.legend(loc='upper right', fancybox=True)

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
