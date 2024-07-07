import os
import logging

import numpy as np

from matplotlib import pyplot as plt

from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit

from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.measurement import Measurement
from rapp.utils import round_to_n, round_to_n_with_uncertainty

logger = logging.getLogger(__name__)

WIDTH_10HZ = 1

FILE_PARAMS = {
    "darkcurrent-range4V-samples40000-sps59.csv": {
        "sps": 59.5,
        "band_stop_freq": [(9.45, 0.1), (9.45 * 3, 0.1), (6, 0.1)],
        "high_pass_freq": None,
        "outliers": None,
        "bins": "quantized"
    },
    "darkcurrent-range2V-samples100000.csv": {
        "sps": 838,
        "sep": "\t",
        "band_stop_freq": [(50, 10), (100, 1), (150, 1)],
        "high_pass_freq": None,
        "outliers": None,
        "bins": "quantized"
    },
    "darkcurrent-range4V-samples100000.csv": {
        "sps": 835,
        "band_stop_freq": None,
        # [
        #    (50 * x, 0.5) for x in range(1, 6)] + [
        #    (131.55, 0.5)] + [
        #   (1 * x, 0.1) for x in range(1, 7)] + [
        #   (1 * x, 0.1) for x in range(8, 14)] + [
        #    (1 * x, 0.1) for x in range(15, 20)] + [
        #    (23, 0.1), (24, 0.1)],
        "high_pass_freq": None,
        "outliers": None,
        # "bins": 35
        "bins": "quantized"
    },
    "continuous-range4V-584nm-samples10000-sps59.csv": {
        "sps": 59.5,
        "band_stop_freq": None,  # [(9.4, 0.3), (18.09, 0.3), (18.8, 0.3), (28.22, 0.3)],
        "high_pass_freq": 2,
        "outliers": None,
        "bins": "quantized"
    },
    "continuous-range4V-632nm-samples100000.csv": {
        "sps": 847,  # This fits OK with line frequencies.
        "band_stop_freq": [
            (50, 15)] + [(50 * x, 2) for x in range(2, 8)] + [
            (110, WIDTH_10HZ), (170, WIDTH_10HZ), (210, WIDTH_10HZ), (230, WIDTH_10HZ),
            (260, WIDTH_10HZ), (270, WIDTH_10HZ), (310, WIDTH_10HZ), (320, WIDTH_10HZ),
            (330, WIDTH_10HZ), (360, WIDTH_10HZ), (380, WIDTH_10HZ)],
        "high_pass_freq": 2,
        "outliers": [-0.001, 0.001],
        "bins": "quantized"
    }
}


def quantized_bins(data, step=0.000125):
    return np.arange(-0.000125 * 20, 0.000125 * 20, step=0.000125)


def generate_quantized_bins(data, step=0.125e-3):
    centers = np.arange(data.min(), data.max() + step, step)
    edges = centers - step / 2
    edges = np.append(edges, max(edges) + step)

    return centers, edges


def linear(f, a, b):
    return - a * f + b


def pink_noise(f, alpha, a):
    return a / f ** alpha


def poly_2(x, A, B, C):
    return A * x**2 + B * x + C


def detrend_poly(data, func):
    popt, pcov = curve_fit(func, np.arange(data.size), data)
    return data - func(np.arange(data.size), *popt)


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

        logger.info("CH{} noise (mu, sigma) = ({}, {})".format(i, mu, sigma))

        pdf_x = np.linspace(min(channel_data), max(channel_data), num=1000)
        pdf_y = stats.norm.pdf(pdf_x, mu, sigma)

        channel_bins = bins[i] if isinstance(bins, list) else bins
        if channel_bins == 'quantized':
            centers, edges = generate_quantized_bins(channel_data, step=0.125e-3)
            counts, edges = np.histogram(channel_data, bins=edges, density=True)
        else:
            counts, edges = np.histogram(channel_data, bins=channel_bins, density=True)
            centers = (edges + np.diff(edges)[0] / 2)[:-1]

        logger.info("CH{} - Amount of bins: {}".format(i, len(centers)))

        width = np.diff(edges)

        import sys
        np.set_printoptions(threshold=sys.maxsize)
        logger.debug("Centers: {}".format(centers))
        logger.debug("Widths: {}".format(width))

        ax.bar(
            centers, counts,
            width=np.diff(edges), color='silver', alpha=0.8, lw=1, edgecolor='k',
            label='Datos'
        )

        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        fit_label = "Gauss PSD\nµ = {:.1E}\nσ = {}".format(mu_rounded, sigma_rounded)

        ax.set_xlabel(ct.LABEL_VOLTAGE)
        ax.set_title("Canal {}".format(i))

        ax.plot(pdf_x, pdf_y, 'k', lw=2, label=fit_label)
        ax.legend(loc='upper left', fontsize=13, frameon=False)
        ax.set_xlim(centers[0] - abs(centers[0]) * 0.5)

        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    f.tight_layout()
    f.subplots_adjust(wspace=0.03)
    f.savefig("{}-histogram.png".format(prefix))


def plot_noise_with_laser_off(output_folder, show=False):
    print("")
    logger.info("PROCESSING SIGNAL WITH LASER OFF (dark current)...")

    # filename = "darkcurrent-range4V-samples40000-sps59.csv"
    # filename = "darkcurrent-range2V-samples100000.csv"
    filename = "darkcurrent-range4V-samples100000.csv"

    sps, bstop, hpass, outliers, bins = FILE_PARAMS[filename].values()
    filepath = os.path.join(ct.INPUT_DIR, filename)

    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    measurement = Measurement.from_file(filepath)

    logger.info("Plotting raw data...")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True)

    for i, ax in enumerate(axs):
        channel_data = measurement.channel_data('CH{}'.format(i))
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
        channel_data = measurement.channel_data('CH{}'.format(i))

        psd = (2 * np.abs(np.fft.fft(channel_data)) ** 2) / (sps * len(channel_data))

        xs = np.arange(0, len(psd))
        xs = (xs / len(psd)) * sps
        N = len(xs)

        end = N // 2
        end = np.where(xs < 10)[0][-1]  # 1/f noise only below 10 Hz

        pink_x = xs[1:end]

        popt, pcov = curve_fit(linear, np.log(pink_x), np.log(psd[1:end]))

        us = np.sqrt(np.diag(pcov))

        alpha, alpha_u = round_to_n_with_uncertainty(popt[0], us[0], n=1, k=1)
        logger.info("A/f^α noise estimation: α = {} ± {}".format(alpha, alpha_u))

        scale, scale_u = round_to_n_with_uncertainty(popt[1], us[1], n=1, k=1)
        logger.info("A/f^α noise estimation: scale = {} ± {}".format(scale, scale_u))

        label = "1/fᵅ (α = {:.2f} ± {:.2f})".format(alpha, alpha_u)

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

        ax.legend(loc='upper left', frameon=False, fontsize=12)

    f.subplots_adjust(wspace=0.03)
    f.tight_layout()
    f.savefig("{}-fft.png".format(base_output_fname))

    # if show:
    #    plt.show()

    plt.close()

    filtered = []
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=True)
    for i, ax in enumerate(axs):
        channel_data = measurement.channel_data('CH{}'.format(i))

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
    channel_data = measurement.channel_data()
    height = max(abs(channel_data.max()))
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

    # filename = "continuous-range4V-584nm-samples10000-sps59.csv"
    filename = "continuous-range4V-632nm-samples100000.csv"

    filepath = os.path.join(ct.INPUT_DIR, filename)

    sps, bstop, hpass, outliers, bins = FILE_PARAMS[filename].values()

    measurement = Measurement.from_file(filepath)
    data = measurement[1:]  # remove big outlier in ch0
    base_output_fname = "{}".format(os.path.join(output_folder, filename[:-4]))

    logger.info("Fitting raw data to 2-degree polynomial...")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=False)
    for i, ax in enumerate(axs):
        channel_data = measurement.channel_data('CH{}'.format(i))[1:]

        logger.info("mean CH{}: {}".format(i, np.mean(channel_data)))
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
        ax.legend(loc='lower left', frameon=False, fontsize=12)

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
    for ch in Measurement.channel_names():
        data[ch] = detrend_poly(data[ch], poly_2)

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

        alpha, alpha_u = round_to_n_with_uncertainty(popt[0], us[0], n=1, k=1)
        logger.info("A/f^α noise estimation: α = {} ± {}".format(alpha, alpha_u))

        scale, scale_u = round_to_n_with_uncertainty(popt[1], us[1], n=1, k=1)
        logger.info("A/f^α noise estimation: scale = {} ± {}".format(scale, scale_u))

        # label = "1/fᵅ (α = {:.2f} ± {:.2f})".format(alpha, alpha_u)
        # pink_y = pink_noise(xs[1:end], alpha, np.exp(popt[1]))[:end]

        if i == 0:
            ax.set_ylabel(ct.LABEL_PSD)

        ax.set_xlabel(ct.LABEL_FREQUENCY)
        ax.set_title("Canal {}".format(i))
        ax.loglog(xs[1:N // 2], psd[1:N // 2], color='k')
        # ax.loglog(pink_x, pink_y, color='deeppink', lw=2, label=label)

        line_frequencies = [50 * x for x in range(1, int(xs[N // 2] / 50))]
        for i, freq in enumerate(line_frequencies, 0):
            freq_label = None
            if i == 0:
                freq_label = "50 Hz y armónicos"
            ax.axvline(x=freq, ls='--', lw=1, label=freq_label)

        ax.legend(loc='lower left', frameon=False, fontsize=12)

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
                b, a = signal.butter(1, [fr - delta, fr + delta], btype='bandstop', fs=sps)
                channel_data = signal.lfilter(b, a, channel_data)

        if hpass is not None:
            logger.info("Filtering highpass: {} Hz".format(hpass))

            b, a = signal.butter(1, hpass, btype='highpass', fs=sps)
            channel_data = signal.filtfilt(b, a, channel_data)

        if outliers is not None:
            channel_data[channel_data < outliers[0]] = 0
            channel_data[channel_data > outliers[1]] = 0

        res = stats.normaltest(channel_data)
        logger.info("Gaussian Test. p-value: {}".format(res.pvalue))

        filtered.append(channel_data)

        psd = np.abs(np.fft.fft(channel_data)) ** 2

        xs = np.arange(0, len(psd))
        xs = (xs / len(psd)) * sps
        N = len(xs)
        ax.loglog(xs[:N // 2], psd[:N // 2], color='k')

    f.subplots_adjust(hspace=0)
    f.tight_layout()
    f.savefig("{}-filtered-fft".format(base_output_fname))

    if show:
        plt.show()

    plt.close()

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, ysci=True, folder=output_folder)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    for i, filtered_channel in enumerate(filtered, 0):
        plot.add_data(filtered_channel, style='-', label='Canal {}'.format(i))

    height = max(abs(data.max()))
    plot._ax.set_ylim(-height / 20, height / 20)
    plot.legend(loc='upper right')

    plot.save("{}-filtered-signal.png".format(filename[:-4]))

    if show:
        plt.show()

    plt.close()

    data = np.array(filtered).T
    # data = data[['CH0', 'CH1']].to_numpy()

    plot_histogram_and_pdf(data, bins=bins, prefix=base_output_fname, show=show)

    if show:
        plt.show()

    plt.close()


def plot_noise_with_signal(show=False):
    print("")
    logger.info("ANALYZING NOISE WITH LASER ON...")

    filename = "2024-03-05-repeatability/hwp0/" \
               "repeatability5-hwp0.0-cycles1-step1.0-samples169-rep4.csv"
    # filename = "2024-04-13-simple-setup-0s-delay/" \
    #            "min-setup2-2-hwp0-cycles1-step1-samples169-rep2.csv"

    filepath = os.path.join(ct.INPUT_DIR, filename)

    measurement = Measurement.from_file(filepath)

    logger.info("Show standard deviation as a function of intensity.")
    f, axs = plt.subplots(1, 2, figsize=(8, 5), subplot_kw=dict(box_aspect=1), sharey=False)

    xs, s1, s2, s1u, s2u = measurement.average_data()

    axs[0].plot(xs, s1, 'o', color='k', mfc='None', ms=4, markevery=5, alpha=0.6, label="Datos")
    axs[0].title.set_text('Averaged signal (1 sample per angle)')

    axs[1].plot(xs, s1u)
    axs[1].title.set_text('Std (1 std value per angle)')

    if show:
        plt.show()

    plt.close()
