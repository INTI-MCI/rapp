import os
import logging

import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.measurement import Measurement
from rapp.utils import create_folder

logger = logging.getLogger(__name__)

COVERAGE_FACTOR = 3


def sine(xs, a, phi, c):
    return a * np.sin(4 * xs + phi) + c


def phase_difference_from_file(filepath, method, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    measurement = Measurement.from_file(filepath)
    logger.info("Parameters: {}.".format(measurement.parameters_string()))

    filename = "{}.png".format(os.path.basename(filepath)[:-4])

    phase_difference(measurement, method, filename=filename, show=show)


def phase_difference(measurement, method, filename=None, show=False):
    xs, s1, s2, s1err, s2err, res = measurement.phase_diff(method=method)

    phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)

    log_phi = "{} (k=1).".format("φ=({} ± {})°".format(phase_diff, phase_diff_u))
    logger.info("Detected phase difference (analyzer angles): {}".format(log_phi))

    if method in ['ODR', 'NLS'] and (filename or show):
        plot_phase_difference((xs, s1, s2, s1err, s2err, res), filename, show)


def plot_phase_difference(phase_diff_result, filename, show=False):
    xs, s1, s2, s1err, s2err, res = phase_diff_result

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # Plot data and sinusoidal fits

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_DEGREE, folder=output_folder)

    markevery = int(len(xs) * 0.02)

    d1 = plot.add_data(
        xs, s1,
        yerr=s1err,
        color='k',
        mew=1,
        markevery=markevery,
        alpha=0.8,
        label='CH0',
        style='D'
    )

    d2 = plot.add_data(
        xs, s2,
        yerr=s2err,
        color='k',
        mew=1,
        markevery=markevery,
        alpha=0.8,
        label='CH1',
    )

    left_legend = [d1, d2]
    right_legend = []

    error = np.sum((res.fits1 - s1) ** 2)
    logger.info("ERROR between CH0 data and Model: {}".format(error))

    error = np.sum((res.fits2 - s2) ** 2)
    logger.info("ERROR between CH1 data and Model: {}".format(error))

    phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)
    label_phi = "φ=({} ± {})°".format(phase_diff, phase_diff_u)

    f1 = plot.add_data(res.fitx, res.fits1, style='-', color='k', lw=1, label=label_phi)
    plot.add_data(res.fitx, res.fits2, style='-', color='k', lw=1)

    signal_diff_s1 = s1 - res.fits1
    signal_diff_s2 = s2 - res.fits2

    l1 = plot.add_data(res.fitx, signal_diff_s1, style='-', lw=1.5, label='CH0 diff')
    l2 = plot.add_data(res.fitx, signal_diff_s2, style='-', lw=1.5, label='CH1 diff')

    right_legend.extend([l1, l2])
    left_legend.append(f1)

    first_legend = plot._ax.legend(handles=left_legend, loc='upper left', frameon=False)
    plot._ax.add_artist(first_legend)
    plot._ax.legend(handles=right_legend, loc='upper right', frameon=False)

    plot._ax.set_ylim(min(s1) - abs(max(s1) - min(s1)) * 0.2, max(s1) * 1.8)

    plot.save(filename)

    # Plot difference between data and fitted model.

    def f1(k1):
        return 1 + k1 * signal_diff_s2

    def f2(xs, phi):
        return np.sin(4 * xs + phi)

    def f3(xs, k2, phi):
        return (k2 + np.sin(2 * xs + np.pi / 4 + phi / 2))

    def residual(xs, A, k1, k2, phi, c):
        return A * f1(k1) * f2(xs, phi) * f3(xs, k2, phi) + c

    p0 = [0.03, 0, 0, 0, 0]

    xs_rad = np.deg2rad(res.fitx)
    popt, pcov = curve_fit(
        residual, xs_rad, signal_diff_s1, sigma=s1err, absolute_sigma=True, p0=p0)

    residual_fitx = xs_rad
    residual_fitx_deg = np.rad2deg(residual_fitx)

    residual_fity = residual(residual_fitx, *popt)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_DEGREE, folder=output_folder)
    plot.add_data(
        res.fitx, signal_diff_s1,
        style='o', color='k', mfc='None', markevery=3, lw=1, label='Diff CH0'
    )
    plot.add_data(res.fitx, signal_diff_s2, style='-', lw=1, label='Diff CH1')

    plot.add_data(
        residual_fitx_deg, residual_fity, style='-', color='k', lw=1, label='Ajuste Diff CH0'
    )

    plt.legend(loc='upper left', frameon=False)
    plot.save(filename="{}-difference.png".format(filename[-4]))

    plot._ax.set_ylim(
        np.min([signal_diff_s1, signal_diff_s2]),
        np.max([signal_diff_s1, signal_diff_s2]) * 1.5)

    # Plot different components of residual model

    f, axs = plt.subplots(4, 1, figsize=(8, 5), sharex=True)
    y0 = f1(popt[1])
    y1 = f2(residual_fitx, popt[3])
    y2 = f3(residual_fitx, popt[2], popt[3])
    y3 = residual_fity

    axs[0].plot(residual_fitx_deg, y0, color='k')
    axs[1].plot(residual_fitx_deg, y1, color='k')
    axs[2].plot(residual_fitx_deg, y2, color='k')
    axs[3].plot(res.fitx, signal_diff_s1, 'o', color='k', ms=5, mfc='None', markevery=5, mew=0.5)
    axs[3].plot(residual_fitx_deg, y3, lw=1, color='k')

    error = np.sum((residual_fity - signal_diff_s1) ** 2)
    logger.info("ERROR between CH0 residual and Model: {}".format(error))

    if show:
        plot.show()

    plot.close()
