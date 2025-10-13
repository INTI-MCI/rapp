import os
import logging
import glob

import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit  # noqa
from scipy.special import gamma
from scipy.signal import hilbert
import matplotlib as mpl

from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.measurement import Measurement, process_temperature_data
from rapp.utils import create_folder, round_to_n, sort_files_by_rep

mpl.rcParams.update(mpl.rcParamsDefault)

logger = logging.getLogger(__name__)

FORMAT = "png"


def sine(xs, a, phi, c):
    return a * np.sin(4 * xs + phi) + c


def phase_difference_from_folder(
    folder, method, norm=False, fill_none=False, appended_measurements=None,
    correlation=False, plot=False, show=False, **kwargs
):
    logger.info("Calculating phase difference for {}...".format(folder))

    files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if x.endswith("csv") and
                    x != "temperature.csv" and x != "qp-temperature.csv"])
    if not files:
        raise ValueError("Folder does not contain measurements!")

    files = sort_files_by_rep(files)

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
            logger.debug("Processing {}...".format(filepath))
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
        logger.debug("STD phase of CH0: {}".format(std_phi1))

    mean_phi2 = None
    std_phi2 = None
    if len(phi2) > 0:
        std_phi2 = np.std(phi2)
        mean_phi2 = np.mean(phi2)
        logger.debug("STD phase of CH1: {}".format(std_phi2))

    mean_phase_diff = np.mean(phase_diffs)
    n = len(phase_diffs)
    std_phase_diff = np.std(phase_diffs, ddof=1) if n > 1 else 0.0
    # Standard deviation of the sample standard deviation.
    # The latter should be np.std(phase_diffs, ddof=1).
    # See https://stats.stackexchange.com/questions/631/standard-deviation-of-standard-deviation
    if n > 1:
        std_std = std_phase_diff * np.sqrt(1 - 2 / (n - 1) * (gamma(n / 2) / gamma((n - 1) / 2))
                                           ** 2)
    else:
        std_std = 0.0

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

    temperature0 = process_temperature_data(folder, filename="temperature")
    temperature1 = process_temperature_data(folder, filename="qp-temperature")

    if correlation:
        correlation_ph_0_temp0 = np.corrcoef(phi1, temperature0[0], rowvar=False)
        correlation_ph_1_temp0 = np.corrcoef(phi2, temperature1[0], rowvar=False)
        correlation_ph_diff_temp0 = np.corrcoef(phase_diffs, temperature0[0], rowvar=False)
        correlation_ph_0_temp1 = np.corrcoef(phi1, temperature1[0], rowvar=False)
        correlation_ph_1_temp1 = np.corrcoef(phi2, temperature1[0], rowvar=False)
        correlation_ph_diff_temp1 = np.corrcoef(phase_diffs, temperature1[0], rowvar=False)
        logger.info("Correlation phase ch 0 and temperature ch 0: {}".format(
            correlation_ph_0_temp0[0, 1]))
        logger.info("Correlation phase ch 1 and temperature ch 0: {}".format(
            correlation_ph_1_temp0[0, 1]))
        logger.info("Correlation phase diffs and temperature ch 0: {}".format(
            correlation_ph_diff_temp0[0, 1]))
        # logger.info("Correlation matrix (phase diff and t ch0): {}".format(correlation_ph_diff_temp0))
        logger.info("Correlation phase ch 0 and temperature ch 1: {}".format(
            correlation_ph_0_temp1[0, 1]))
        logger.info("Correlation phase ch 1 and temperature ch 1: {}".format(
            correlation_ph_1_temp1[0, 1]))
        logger.info("Correlation phase diffs and temperature ch 1: {}".format(
            correlation_ph_diff_temp1[0, 1]))
        # logger.info("Correlation matrix (phase diff and t ch1): {}".format(correlation_ph_diff_temp1))

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

        label_phi1 = "N/D" if std_phi1 is None else "{}°".format(round_to_n(std_phi1, 2))
        axs[0].plot(phi1, "-", color="k", label="STD = {}".format(label_phi1))
        axs[0].set_title("CH0")
        axs[0].set_ylabel("Fase intrínseca (°)")
        axs[0].set_xlabel("Nro de repetición")
        axs[0].legend()

        label_phi2 = "N/D" if std_phi2 is None else "{}°".format(round_to_n(std_phi2, 2))
        axs[1].plot(phi2, "-", color="k", label="STD = {}".format(label_phi2))
        axs[1].set_ylabel("Fase intrínseca (°)")
        axs[1].set_xlabel("Nro de repetición")
        axs[1].set_title("CH1")
        axs[1].legend()

        label_phase_diff = "Diferencia de fase"
        axs[2].plot(phase_diffs, ".-", color="k", label=label_phase_diff)
        axs[2].set_ylabel("Diferencia de fase (°)")
        axs[2].set_xlabel("Nro de repetición")
        axs[2].set_title("DIFF")
        axs[2].legend()
        twin2 = axs[2].twinx()
        twin2.plot(temperature0[0], linestyle="-", color="r", label="Temperatura Media 0")
        twin2.set_ylabel("Temperatura (°C)")
        twin2.plot(temperature0[1], linestyle=":", color="r", label="Temperatura Max/Min 0")
        twin2.plot(temperature0[2], linestyle=":", color="r")
        twin2.plot(temperature1[0], linestyle="-", color="b", label="Temperatura Media 1")
        twin2.set_ylabel("Temperatura (°C)")
        twin2.plot(temperature1[1], linestyle=":", color="b", label="Temperatura Max/Min 1")
        twin2.plot(temperature1[2], linestyle=":", color="b")
        twin2.legend()

        f.tight_layout()

        filename = os.path.join(output_folder, "difference-vs-time.{}".format(FORMAT))
        f.savefig(fname=filename)

        counts, edges = np.histogram(phase_diffs, density=True)
        centers = (edges + np.diff(edges)[0] / 2)[:-1]

        plot = Plot(ylabel="Cuentas", xlabel="Diferencia de fase", folder=output_folder)
        plot.the_ax.bar(
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
# TODO: agregar barras de error a las diferencias de fase
        plt.figure()  # Plot for presentation, phase diff and temperature vs reps
        plt.plot(phase_diffs, ".-", color="k", label=label_phase_diff)
        plt.ylabel("Diferencia de fase / °", size=14)
        plt.xlabel("Nº de repetición", size=14)
        plt.title("Diferencia de fase con placa", size=17)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        twin2 = plt.twinx()
        twin2.plot(temperature0[0], linestyle="-", color="r", label="Temperatura Media 0")
        twin2.plot(temperature0[1], linestyle=":", color="r", label="Temperatura Max/Min 0")
        twin2.plot(temperature0[2], linestyle=":", color="r")
        twin2.plot(temperature1[0], linestyle="-", color="b", label="Temperatura Media 1")
        twin2.plot(temperature1[1], linestyle=":", color="b", label="Temperatura Max/Min 1")
        twin2.plot(temperature1[2], linestyle=":", color="b")
        twin2.set_ylabel("Temperatura / °C", size=14)
        twin2.legend(fontsize=14)
        twin2.tick_params(axis='y', labelsize=12)

        plt.tight_layout()

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

    phase_difference(measurement, method, norm=norm, filename=filename, show=show, **kwargs)


def phase_difference(
    measurement: Measurement, method, filename=None, norm=False, show=False, normplot=False,
    **kwargs
):
    xs, s1, s2, s1err, s2err, res = measurement.phase_diff(method=method, norm=norm, **kwargs)

    logger.debug("Minimum of CH0 signal: {}".format(min(s1)))
    logger.debug("Minimum of CH1 signal: {}".format(min(s2)))

    phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)

    log_phi = "{} (k=1).".format("φ=({} ± {})°".format(phase_diff, phase_diff_u))
    logger.debug("Detected phase difference (analyzer angles): {}".format(log_phi))

    if method in ["ODR", "NLS", "WNLS", "DFT", "ANNEAL"] and (filename or show):
        plot_phase_difference((xs, s1, s2, s1err, s2err, res), filename=filename, show=show,
                              norm=normplot)

    return (xs, s1, s2, s1err, s2err, res)


def plot_phase_difference(phase_diff_result, work_dir=ct.WORK_DIR, filename=None, show=False,
                          norm=False):
    xs, s1, s2, s1err, s2err, res = phase_diff_result

    if norm:
        s1max, s2max = s1.max(), s2.max()
        s1 /= s1max
        s2 /= s2max
        s1err /= s1max
        s2err /= s2max
        res.fits1 /= s1max
        res.fits2 /= s2max

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # Plot data and sinusoidal fits

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=output_folder)

    markevery = int(len(xs) * 0.02)
    # plot.the_ax.set_xlim(140, 360)

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

        first_legend = plot.the_ax.legend(handles=left_legend, loc="upper left", frameon=False)
        plot.the_ax.add_artist(first_legend)
        plot.the_ax.legend(handles=right_legend, loc="upper right", frameon=False)

        plot.the_ax.set_ylim(min(s1) - abs(max(s1) - min(s1)) * 0.2, max(s1) * 1.05)

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
        plot.the_ax.set_ylim(minimum - abs(minimum) * 0.1, 0.06)
        plot.move((0, 50))

    if show:
        plot.show()

    plot.close()


def instantaneous_phase_difference(
    filepath, norm=False, fill_none=False, plot=False, show=False, **kwargs
):
    folders = [os.path.join(filepath, f) for f in os.listdir(filepath) if os.path.isdir(
        os.path.join(filepath, f))]
    if len(folders) != 2:
        ValueError("Folder {} does not contain two folders.".format(filepath))
    files_i = glob.glob(f"{folders[0]}/*.csv")
    files_i = [f for f in files_i if not f.endswith("temperature.csv") and not f.endswith(
        "qp-temperature.csv")]
    files_i = sorted(files_i)
    files_f = glob.glob(f"{folders[1]}/*.csv")
    files_f = [f for f in files_f if not f.endswith("temperature.csv") and not f.endswith(
        "qp-temperature.csv")]
    files_f = sorted(files_f)

    for file_i, file_f in zip(files_i, files_f):
        measurement_i = Measurement.from_file(file_i, fill_none=fill_none)
        measurement_f = Measurement.from_file(file_f, fill_none=fill_none)

        xs_i, s1_i, s2_i, s1_sigma_i, s2_sigma_i = measurement_i.average_data(norm=norm)
        xs_f, s1_f, s2_f, s1_sigma_f, s2_sigma_f = measurement_f.average_data(norm=norm)

        analytic_s1_i = hilbert(s1_i - s1_i.mean())
        inst_phase1_i = np.angle(analytic_s1_i)
        analytic_s2_i = hilbert(s2_i - s2_i.mean())
        inst_phase2_i = np.angle(analytic_s2_i)
        inst_phase_diff_i = np.exp(1j * inst_phase2_i) / np.exp(1j * inst_phase1_i)
        inst_phase_diff_i = np.angle(inst_phase_diff_i)

        analytic_s1_f = hilbert(s1_f - s1_f.mean())
        inst_phase1_f = np.angle(analytic_s1_f)
        analytic_s2_f = hilbert(s2_f - s2_f.mean())
        inst_phase2_f = np.angle(analytic_s2_f)
        inst_phase_diff_f = np.exp(1j * inst_phase2_f) / np.exp(1j * inst_phase1_f)
        inst_phase_diff_f = np.angle(inst_phase_diff_f)

        fig, axs = plt.subplots(1, 3)
        axs[0].plot(xs_i, s1_i, 'r', label="CH0-NoQuartz")
        axs[0].plot(xs_i, s2_i, 'r--', label="CH1-NoQuartz")
        axs[0].plot(xs_i, s1_f, 'k', label="CH0-Quartz")
        axs[0].plot(xs_i, s2_f, 'k--', label="CH1-Quartz")
        axs[0].legend(loc="upper left", frameon=False)
        axs[1].plot(xs_i, np.rad2deg(inst_phase_diff_f - inst_phase_diff_i))
        axs[1].set_title("Phase difference")
        axs[2].plot(xs_i, inst_phase_diff_i - inst_phase_diff_i.mean())
        axs[2].plot(xs_i, inst_phase_diff_f - inst_phase_diff_f.mean())
        axs[2].set_title("Phase difference (no mean)")

        unwrap_inst_phase1_i = np.unwrap(inst_phase1_i)
        unwrap_inst_phase2_i = np.unwrap(inst_phase2_i)
        coef_lin_1i = np.polyfit(xs_i, unwrap_inst_phase1_i, deg=1)
        coef_lin_2i = np.polyfit(xs_i, unwrap_inst_phase2_i, deg=1)
        lin_1i = np.polyval(coef_lin_1i, xs_i)
        lin_2i = np.polyval(coef_lin_2i, xs_i)
        const_inst_phase1_i = unwrap_inst_phase1_i - lin_1i
        const_inst_phase2_i = unwrap_inst_phase2_i - lin_2i

        unwrap_inst_phase1_f = np.unwrap(inst_phase1_f)
        unwrap_inst_phase2_f = np.unwrap(inst_phase2_f)
        coef_lin_1f = np.polyfit(xs_f, unwrap_inst_phase1_f, deg=1)
        coef_lin_2f = np.polyfit(xs_f, unwrap_inst_phase2_f, deg=1)
        lin_1f = np.polyval(coef_lin_1f, xs_f)
        lin_2f = np.polyval(coef_lin_2f, xs_f)
        const_inst_phase1_f = unwrap_inst_phase1_f - lin_1f
        const_inst_phase2_f = unwrap_inst_phase2_f - lin_2f

        inst_diff_objeto = np.angle(np.exp(1j * inst_phase2_f) / np.exp(1j * inst_phase2_i))

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(xs_i, const_inst_phase1_i, 'r', label="CH0-NoQuartz")
        axs[0].plot(xs_i, const_inst_phase2_i, 'r--', label="CH1-NoQuartz")
        axs[0].plot(xs_i, const_inst_phase1_f, 'k', label="CH0-Quartz")
        axs[0].plot(xs_i, const_inst_phase2_f, 'k--', label="CH1-Quartz")
        axs[0].legend(loc="upper left", frameon=False)
        axs[1].plot(xs_i, inst_diff_objeto)
        axs[1].set_title("Phase difference objeto")
        plt.show()
