import os
import logging

import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit  # noqa
from scipy.special import gamma


from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.measurement import Measurement
from rapp.utils import create_folder, round_to_n

logger = logging.getLogger(__name__)

FORMAT = "png"


def sine(xs, a, phi, c):
    return a * np.sin(4 * xs + phi) + c


def phase_difference_from_folder(
    folder, method, norm=False, fill_none=False, appended_measurements=None, plot=False,
    show=False, **kwargs
):
    logger.info("Calculating phase difference for {}...".format(folder))

    files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if x.endswith("csv")])
    if not files:
        raise ValueError("Folder does not contain measurements!")

    results = []

    for file_number, filepath in enumerate(files):
        if appended_measurements is None or file_number % appended_measurements == 0:
            measurement = Measurement.from_file(filepath, fill_none=fill_none)
        else:
            measurement.append(Measurement.from_file(filepath, fill_none=fill_none), degrees=True)

        new_measurement = (
            appended_measurements is None or (file_number + 1) % appended_measurements == 0
        )

        # logger.info("Parameters: {}.".format(measurement.parameters_string()))
        if new_measurement:
            res = phase_difference(measurement, method, norm=norm, show=False, **kwargs)
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
            mse = (np.sum(signal_diff_s1**2) + np.sum(signal_diff_s2**2)) / (len(s1) * 2)
            mses.append(mse)

            if mse > 0.002:
                logger.info("Outlier")
                logger.info(mse)
                logger.info("Repetition: {}".format(i))

    mean_phi1 = None
    std_phi1 = None
    if len(phi1) > 0:
        std_phi1 = np.std(phi1)
        mean_phi1 = np.mean(phi1)
        logger.info("STD phase of CH0: {}".format(std_phi1))

    mean_phi2 = None
    std_phi2 = None
    if len(phi2) > 0:
        std_phi2 = np.std(phi2)
        mean_phi2 = np.mean(phi2)
        logger.info("STD phase of CH1: {}".format(std_phi2))

    mean_phase_diff = np.mean(phase_diffs)
    std_phase_diff = np.std(phase_diffs, ddof=1)
    # Standard deviation of the sample standard deviation.
    # The latter should be np.std(phase_diffs, ddof=1).
    # See https://stats.stackexchange.com/questions/631/standard-deviation-of-standard-deviation
    n = len(phase_diffs)
    std_std = std_phase_diff * np.sqrt(1 - 2 / (n - 1) * (gamma(n / 2) / gamma((n - 1) / 2)) ** 2)

    logger.info("Mean phase difference: {}".format(mean_phase_diff))
    logger.info("STD phase difference: {:.5f} ± {:.5f} (k=1)".format(std_phase_diff, std_std))

    reps = len(files)

    row = [
        measurement._cycles,
        measurement._step,
        measurement._samples,
        reps,
        mean_phi1,
        std_phi1,
        mean_phi2,
        std_phi2,
        mean_phase_diff,
        std_phase_diff,
        std_std,
    ]

    if plot or show:
        output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
        f, axs = plt.subplots(
            1,
            3,
            figsize=(12, 4),
            # subplot_kw=dict(box_aspect=1),
            sharey=False,
            sharex=True,
        )

        axs[0].plot(phi1, "-", color="k", label="STD = {}°".format(round_to_n(std_phi1, 2)))
        axs[0].set_title("CH0")
        axs[0].set_ylabel("Fase intrínseca (°)")
        axs[0].set_xlabel("Nro de repetición")
        axs[0].legend()

        axs[1].plot(phi2, "-", color="k", label="STD = {}°".format(round_to_n(std_phi2, 2)))
        axs[1].set_ylabel("Fase intrínseca (°)")
        axs[1].set_xlabel("Nro de repetición")
        axs[1].set_title("CH1")
        axs[1].legend()

        label_phase_diff = "STD = {}°".format(round_to_n(std_phase_diff, 2))
        axs[2].plot(phase_diffs, ".-", color="k", label=label_phase_diff)
        axs[2].set_ylabel("Diferencia de fase (°)")
        axs[2].set_xlabel("Nro de repetición")
        axs[2].set_title("DIFF")
        axs[2].legend()

        f.tight_layout()

        filename = os.path.join(output_folder, "difference-vs-time.{}".format(FORMAT))
        f.savefig(fname=filename)

        counts, edges = np.histogram(phase_diffs, density=True)
        centers = (edges + np.diff(edges)[0] / 2)[:-1]

        plot = Plot(ylabel="Cuentas", xlabel="Diferencia de fase", folder=output_folder)
        plot._ax.bar(
            centers,
            counts,
            width=np.diff(edges),
            color="silver",
            alpha=0.8,
            lw=1,
            edgecolor="k",
            label="Diferencia de fase",
        )
        plot.save(filename="phase-difference-histogram.{}".format(FORMAT))
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

    return row


def phase_difference_from_file(
    filepath, method, norm=False, fill_none=False, plot=False, show=False, **kwargs
):
    logger.info("Calculating phase difference for {}...".format(filepath))

    measurement = Measurement.from_file(filepath, fill_none=fill_none)
    logger.info("Parameters: {}.".format(measurement.parameters_string()))

    filename = None
    if plot or show:
        filename = "{}.{}".format(os.path.basename(filepath)[:-4], FORMAT)

    phase_difference(measurement, method, filename=filename, show=show)


def phase_difference(
    measurement: Measurement, method, filename=None, norm=False, show=False, **kwargs
):
    xs, s1, s2, s1err, s2err, res = measurement.phase_diff(method=method, norm=norm, **kwargs)

    logger.info("Minimum of CH0 signal: {}".format(min(s1)))
    logger.info("Minimum of CH1 signal: {}".format(min(s2)))

    phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)

    log_phi = "{} (k=1).".format("φ=({} ± {})°".format(phase_diff, phase_diff_u))
    logger.info("Detected phase difference (analyzer angles): {}".format(log_phi))

    if method in ["ODR", "NLS", "WNLS", "DFT", "ANNEAL"] and (filename or show):
        plot_phase_difference((xs, s1, s2, s1err, s2err, res), filename=filename, show=show)

    return (xs, s1, s2, s1err, s2err, res)


def plot_phase_difference(phase_diff_result, work_dir=ct.WORK_DIR, filename=None, show=False):
    xs, s1, s2, s1err, s2err, res = phase_diff_result

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # Plot data and sinusoidal fits

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=output_folder)

    markevery = int(len(xs) * 0.02)
    plot._ax.set_xlim(140, 360)

    d1 = plot.add_data(
        xs,
        s1,
        yerr=s1err,
        color="k",
        mew=1,
        markevery=markevery,
        alpha=0.8,
        label="CH0",
        style="D",
        fillstyle="bottom"
    )

    d2 = plot.add_data(
        xs,
        s2,
        yerr=s2err,
        color="k",
        mew=1,
        markevery=markevery,
        alpha=0.8,
        label="CH1",
        mfc="None",
    )

    left_legend = [d1, d2]
    right_legend = []

    if res.fits1 is not None:
        ch0_relative_error = np.sqrt(np.sum((res.fits1 - s1) ** 2) / np.sum(s1**2))
        logger.info("RMSE (relative) between CH0 data and Model: {}".format(ch0_relative_error))

        ch1_relative_error = np.sqrt(np.sum((res.fits2 - s2) ** 2) / np.sum(s2**2))
        logger.info("RMSE (relative) between CH1 data and Model: {}".format(ch1_relative_error))

        phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)
        label_fit = "Fitting method"

        f1 = plot.add_data(res.fitx, res.fits1, style="-", color="k", lw=1, label=label_fit)
        plot.add_data(res.fitx, res.fits2, style="-", color="k", lw=1)

        signal_diff_s1 = s1 - res.fits1
        signal_diff_s2 = s2 - res.fits2

        l1 = plot.add_data(res.fitx, signal_diff_s1, style="-", lw=1.5, label="CH0 diff")
        l2 = plot.add_data(res.fitx, signal_diff_s2, style="-", lw=1.5, label="CH1 diff")

        right_legend.extend([l1, l2])
        left_legend.append(f1)

        first_legend = plot._ax.legend(handles=left_legend, loc="upper left", frameon=False)
        plot._ax.add_artist(first_legend)
        plot._ax.legend(handles=right_legend, loc="upper right", frameon=False)

        plot._ax.set_ylim(min(s1) - abs(max(s1) - min(s1)) * 0.2, max(s1) * 1.05)

        if filename is not None:
            plot.save(filename)

        # Plot difference between data and fitted model.

        phi = np.deg2rad(res.phi1)

        def f1(k1):
            return 1 + k1 * signal_diff_s2

        def f2(xs, phi):
            return np.sin(4 * xs + 2 * phi)

        def f3(xs, k2, phi):
            return k2 + np.sin(2 * xs + np.pi / 4 + phi)

        def residual(xs, A, k1, k2, c):
            return A * f1(k1) * f2(xs, -phi) * f3(xs, k2, -phi) + c

        plot = Plot(
            ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE,
            ysci=True, yoom=-2, folder=output_folder)

        plot.add_data(res.fitx, signal_diff_s1, style=".-", lw=1.5, label="CH0 diff")
        plot.add_data(res.fitx, signal_diff_s2, style=".-", lw=1.5, label="CH1 diff")

        plt.legend(loc="upper left", frameon=False)

        if filename is not None:
            plot.save(filename="{}-residual.{}".format(filename[:-4], FORMAT))

        minimum = np.min([signal_diff_s1, signal_diff_s2])
        plot._ax.set_ylim(minimum - abs(minimum) * 0.1, 0.06)
        plot.move((0, 50))

    if show:
        plot.show()

    plot.close()
