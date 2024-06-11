import os
import logging

import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma


from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.measurement import Measurement
from rapp.utils import create_folder, round_to_n

logger = logging.getLogger(__name__)

COVERAGE_FACTOR = 3


def sine(xs, a, phi, c):
    return a * np.sin(4 * xs + phi) + c


def phase_difference_from_folder(
    folder, method, norm=False, show=False, fill_none=False, appended_measurements=None
):
    logger.info("Calculating phase difference for {}...".format(folder))

    files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if x.endswith('csv')])
    if not files:
        raise ValueError("Folder does not contain measurements!")

    # files = files[1:]  # First measurement is broken in data/2024-03-05-repeatability/hwp0
    # files = files[:-1]   # Last measurement is broken in data/2024-04-05-simple-setup

    results = []

    for file_number, filepath in enumerate(files):
        if appended_measurements is None or file_number % appended_measurements == 0:
            measurement = Measurement.from_file(filepath, fill_none=fill_none)
        else:
            measurement.append(Measurement.from_file(filepath, fill_none=fill_none), degrees=True)

        new_measurement = appended_measurements is None or (
            file_number + 1) % appended_measurements == 0

        # logger.info("Parameters: {}.".format(measurement.parameters_string()))
        if new_measurement:
            res = phase_difference(measurement, method, norm=norm, show=False)
            results.append(res)

    phase_diffs = []
    uncertainties = []
    mses = []
    phi1 = []
    phi2 = []

    for i, res in enumerate(results, 1):
        xs, s1, s2, s1err, s2err, phase_diff = res

        phase_diffs.append(phase_diff.value)
        uncertainties.append(phase_diff.u)

        if phase_diff.phi1 is not None:
            phi1.append(phase_diff.phi1)

        if phase_diff.phi2 is not None:
            phi2.append(phase_diff.phi2)

        if phase_diff.fits1 is not None:
            signal_diff_s1 = s1 - phase_diff.fits1
            signal_diff_s2 = s2 - phase_diff.fits2
            mse = (np.sum(signal_diff_s1 ** 2) + np.sum(signal_diff_s2 ** 2)) / (len(s1) * 2)
            mses.append(mse)

            if mse > 0.002:
                logger.info("Outlier")
                logger.info(mse)
                logger.info("Repetition: {}".format(i))

    if len(phi1) > 0:
        std_phi1 = np.std(phi1)
        logger.info("STD phase of CH0: {}".format(std_phi1))

    if len(phi2) > 0:
        std_phi2 = np.std(phi2)
        logger.info("STD phase of CH1: {}".format(std_phi2))

    std_phase_diffs = np.std(phase_diffs, ddof=1)
    # Standard deviation of the sample standard deviation.
    # The latter should be np.std(phase_diffs, ddof=1).
    # See https://stats.stackexchange.com/questions/631/standard-deviation-of-standard-deviation
    n = len(phase_diffs)
    std_std = std_phase_diffs * np.sqrt(1 - 2/(n - 1) * (gamma(n/2) / gamma((n-1)/2))**2)

    logger.info("STD phase difference: {:.5f} ± {:.5f} (k=1)".format(std_phase_diffs, std_std))

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)

    plot = Plot(ylabel="Fase intrínseca CH0 (°)", xlabel="Nro de repetición",
                folder=output_folder)
    plot.add_data(phi1, style='.-', color='k', label='STD = {}°'.format(
        round_to_n(std_phi1, 2)))
    plot.legend()
    plot.save(filename="phase-CH0-vs-time.png")

    plot = Plot(ylabel="Fase intrínseca CH1 (°)", xlabel="Nro de repetición",
                folder=output_folder)
    plot.add_data(phi2, style='.-', color='k', label='STD = {}°'.format(
        round_to_n(std_phi2, 2)))
    plot.legend()
    plot.save(filename="phase-CH1-vs-time.png")

    plot = Plot(ylabel="Diferencia de fase (°)", xlabel="Nro de repetición",
                folder=output_folder)
    plot.add_data(phase_diffs, style='.-', color='k')
    plot.save(filename="difference-vs-time.png")

    counts, edges = np.histogram(phase_diffs, density=True)
    centers = (edges + np.diff(edges)[0] / 2)[:-1]

    plot = Plot(ylabel="Cuentas", xlabel="Diferencia de fase", folder=output_folder)
    plot._ax.bar(
        centers, counts,
        width=np.diff(edges), color='silver', alpha=0.8, lw=1, edgecolor='k',
        label='Diferencia de fase'
    )
    plot.save(filename="phase-difference-histogram.png")
    plot.close()

    """
    errors = abs(phase_diffs - np.mean(phase_diffs))
    plot = Plot(ylabel="Error (°)", xlabel="Incertidumbre (°)", folder=output_folder)
    plot.add_data(uncertainties, errors, color='k', alpha=0.7)
    plot.save(filename="errors-vs-uncertainties.png")
    plot.close()

    if mses:
        plot = Plot(ylabel="Error (°)", xlabel="MSE (°)", folder=output_folder)
        plot.add_data(mses, errors, color='k', alpha=0.7)
        plot.save(filename="errors-vs-mse.png")
        plot.close()
    """
    if show:
        plot.show()


def phase_difference_from_file(filepath, method, norm=False, fill_none=False, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    measurement = Measurement.from_file(filepath, fill_none=fill_none)
    logger.info("Parameters: {}.".format(measurement.parameters_string()))

    filename = "{}.svg".format(os.path.basename(filepath)[:-4])

    phase_difference(measurement, method, filename=filename, show=show)


def phase_difference(
    measurement: Measurement, method, filename=None, norm=False, show=False, **kwargs
):
    xs, s1, s2, s1err, s2err, res = measurement.phase_diff(method=method, norm=norm, **kwargs)

    phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)

    log_phi = "{} (k=1).".format("φ=({} ± {})°".format(phase_diff, phase_diff_u))
    logger.info("Detected phase difference (analyzer angles): {}".format(log_phi))

    if method in ['ODR', 'NLS', 'WNLS', 'DFT'] and (filename or show):
        plot_phase_difference((xs, s1, s2, s1err, s2err, res), filename, show)

    return (xs, s1, s2, s1err, s2err, res)


def plot_phase_difference(phase_diff_result, work_dir=ct.WORK_DIR, filename=None, show=False):
    xs, s1, s2, s1err, s2err, res = phase_diff_result

    output_folder = os.path.join(work_dir, ct.OUTPUT_FOLDER_PLOTS)
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
        style='D',
        mfc='None',
    )

    d2 = plot.add_data(
        xs, s2,
        yerr=s2err,
        color='k',
        mew=1,
        markevery=markevery,
        alpha=0.8,
        label='CH1',
        mfc='None',
    )

    left_legend = [d1, d2]
    right_legend = []

    if res.fits1 is not None:
        relatove_error = np.sqrt(np.sum((res.fits1 - s1) ** 2) / np.sum(s1 ** 2))
        logger.info("RMSE (relative) between CH0 data and Model: {}".format(relatove_error))

        error = np.sqrt(np.sum((res.fits2 - s2) ** 2) / np.sum(s2 ** 2))
        logger.info("RMSE (relative) between CH1 data and Model: {}".format(error))

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

        if filename is not None:
            plot.save(filename)

        # Plot difference between data and fitted model.

        phi = np.deg2rad(res.phi1)

        def f1(k1):
            return 1 + k1 * signal_diff_s2

        def f2(xs, phi):
            return np.sin(4 * xs + 2 * phi)

        def f3(xs, k2, phi):
            return (k2 + np.sin(2 * xs + np.pi / 4 + phi))

        def residual(xs, A, k1, k2, c):
            return A * f1(k1) * f2(xs, -phi) * f3(xs, k2, -phi) + c

        p0 = [0.03, 0, 0, 0]

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

        if filename is not None:
            plot.save(filename="{}-difference.svg".format(filename[-4]))

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
        axs[3].plot(
            res.fitx, signal_diff_s1, 'o', color='k', ms=5, mfc='None', markevery=5, mew=0.5)
        axs[3].plot(residual_fitx_deg, y3, lw=1, color='k')

        error = np.sqrt(
            np.sum((residual_fity - signal_diff_s1) ** 2) / np.sum(signal_diff_s1 ** 2))
        logger.info("RMSE (relative) between CH0 residual and Model: {}".format(error))

    if show:
        plot.show()

    plot.close()
