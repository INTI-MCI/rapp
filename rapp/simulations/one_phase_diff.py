import os
import logging
import decimal

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot
from rapp.utils import round_to_n, create_folder
from rapp.measurement import Measurement


logger = logging.getLogger(__name__)

TPL_LABEL = "step={}° deg.\ncycles={}.\nφ = {}°.\n"
TPL_FILENAME = "sim_phase_diff_fit-samples-{}-step-{}.png"


def run(phi, folder, method='ODR', samples=50, step=1, cycles=2, show=False):
    print("")
    logger.info("SIGNAL AND PHASE DIFFERENCE SIMULATION")

    fc = simulator.samples_per_cycle(step=step)

    logger.info("Simulating signals...")
    measurement = Measurement.simulate(
        cycles=cycles,
        fc=fc,
        phi=phi,
        fa=samples,
    )

    logger.info("Calculating phase difference...")
    xs, s1, s2, s1_sigma, s2_sigma, res = measurement.phase_diff(
        method=method,
        p0=[1, 0, 0, 0, 0, 0]
    )

    phase_diff_u_rounded = round_to_n(res.u, 2)

    # Obtain number of decimal places of the u:
    d = abs(decimal.Decimal(str(phase_diff_u_rounded)).as_tuple().exponent)

    phase_diff_rounded = round(res.value, d)

    phi_label = "φ=({} ± {})°.".format(phase_diff_rounded, phase_diff_u_rounded)
    title = "cycles={}, step={}, samples={}.".format(cycles, step, samples)

    logger.info(phi_label)
    logger.info(title)

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_RADIANS, folder=output_folder)
    markevery = 10
    d1 = plot.add_data(
        xs * 2, s1, yerr=s1_sigma,
        ms=6, mfc='None', color='k', mew=1,
        markevery=markevery, alpha=0.8, label='CH0',
        style='D',
        xrad=True
    )

    d2 = plot.add_data(
        xs * 2, s2, yerr=s2_sigma,
        ms=6, mfc='None', color='k', mew=1, markevery=markevery, alpha=0.8, label='CH1',
        xrad=True
    )

    first_legend = plot._ax.legend(handles=[d1, d2], loc='upper left', frameon=False)

    # Add the legend manually to the Axes.
    plot._ax.add_artist(first_legend)

    if res.fitx is not None:
        f1 = plot.add_data(
            res.fitx, res.fits1, style='-', color='k', lw=1, label='Ajuste', xrad=True)

        plot.add_data(res.fitx, res.fits2, style='-', color='k', lw=1, xrad=True)

        signal_diff_s1 = s1 - res.fits1
        signal_diff_s2 = s2 - res.fits2

        l1 = plot.add_data(
            res.fitx, signal_diff_s1, style='-', lw=1.5, label='Ajuste - CH0', xrad=True)
        l2 = plot.add_data(
            res.fitx, signal_diff_s2, style='-', lw=1.5, label='Ajuste - CH1', xrad=True)

        plot._ax.legend(handles=[f1, l1, l2], loc='upper right', frameon=False)

    # plot._ax.set_xlim(min(xs) - (max(xs) - min(xs)) * 0.5)
    plot._ax.set_ylim(min(s1) - abs(max(s1) - min(s1)) * 0.2, max(s1) * 1.8)

    plot.save(filename="sim-phase-diff.png")

    if res.fitx is not None:
        plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_RADIANS, folder=output_folder)
        plot.add_data(res.fitx, signal_diff_s1, style='-', lw=1.5, label='Ajuste - CH0')
        plot.add_data(res.fitx, signal_diff_s2, style='-', lw=1.5, label='Ajuste - CH1')
        plot.legend(loc='upper left')
        plot.save(filename="sim-phase-diff-difference.png")
        plot._ax.set_ylim(
            np.min([signal_diff_s1, signal_diff_s2]),
            np.max([signal_diff_s1, signal_diff_s2]) * 1.5)

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
