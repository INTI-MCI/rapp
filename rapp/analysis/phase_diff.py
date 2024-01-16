import os
import logging

import numpy as np

from matplotlib import pyplot as plt

from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.measurement import Measurement
from rapp.utils import create_folder

logger = logging.getLogger(__name__)

COVERAGE_FACTOR = 3


def plot_phase_difference_from_file(filepath, method, show=False):
    logger.info("Calculating phase difference for {}...".format(filepath))

    measurement = Measurement.from_file(filepath)
    logger.info("Parameters: {}.".format(measurement.parameters_string()))

    filename = "{}.png".format(os.path.basename(filepath)[:-4])
    plot_phase_difference(measurement, method, filename, show)


def plot_phase_difference(measurement, method, filename, show=False):
    xs, s1, s2, s1err, s2err, res = measurement.phase_diff(method=method)

    phase_diff, phase_diff_u = res.round_to_n(n=2, k=1)

    label_phi = "φ=({} ± {})°".format(phase_diff, phase_diff_u)
    log_phi = "{} (k=1).".format(label_phi)
    logger.info("Detected phase difference (analyzer angles): {}".format(log_phi))

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_DEGREE, folder=output_folder)

    markevery = int(len(xs) * 0.03)
    d1 = plot.add_data(
        xs, s1,
        yerr=s1err,
        ms=6,
        mfc='None',
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
        ms=6,
        mfc='None',
        color='k',
        mew=1,
        markevery=markevery,
        alpha=0.8,
        label='CH1',
    )

    # plot._ax.text(0.085, 0.73, log_phi, transform=plot._ax.transAxes)

    left_legend = [d1, d2]
    right_legend = []

    if res.fitx is not None:
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

    if res.fitx is not None:
        plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_DEGREE, folder=output_folder)
        plot.add_data(res.fitx, signal_diff_s1, style='-', lw=1.5, label='Ajuste - CH0')
        plot.add_data(res.fitx, signal_diff_s2, style='-', lw=1.5, label='Ajuste - CH1')
        plt.legend(loc='upper left', frameon=False)
        plot.save(filename="{}-difference.png".format(filename[-4]))

        plot._ax.set_ylim(
            np.min([signal_diff_s1, signal_diff_s2]),
            np.max([signal_diff_s1, signal_diff_s2]) * 1.5)

    if show:
        plot.show()

    plot.close()
