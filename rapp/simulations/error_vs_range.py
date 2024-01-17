import logging

import numpy as np

from rapp import constants as ct
from rapp.adc import GAINS
from rapp.simulations import simulation
from rapp.analysis.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "A={}, φerr: {}."
TPL_LABEL = "cycles={}\nsamples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_range-reps-{}-samples-{}-step-{}.png"

np.random.seed(1)


def run(
    phi, folder,
    method='ODR', samples=5, step=0.01, reps=1, cycles=2, gains=None, show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS MAX TENSION")

    if gains is None:
        gains = GAINS

    errors = {}
    percentages = {}
    for max_v, step_mV in gains.values():
        logger.info("MAX V={}, step={}".format(max_v, step))
        v_step = step_mV * 2
        max_A = max_v / 2

        amplitudes = np.linspace(max_A - v_step * 8, max_A, num=8)

        errors[max_v] = []
        for amplitude in amplitudes:
            fc = simulation.samples_per_cycle(step=step)

            n_results = simulation.n_simulations(
                phi=phi,
                N=reps,
                A=amplitude,
                max_v=max_v,
                method='ODR',
                cycles=cycles,
                fc=fc,
                fa=samples,
                allow_nan=True,
                p0=[amplitude, 0, 0, phi, 0, 0]
            )

            error = n_results.rmse()
            errors[max_v].append(error)

            logger.info(TPL_LOG.format(amplitude, "{:.2E}".format(error)))

        percentages[max_v] = ((amplitudes * 2) / max_v) * 100

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_DYNAMIC_RANGE_USE,
        ysci=True,
        xint=False,
        folder=folder
    )

    for max_v in errors:
        label = "{} V".format(max_v)
        plot.add_data(percentages[max_v], errors[max_v], ms=1, style='s-', lw=2, label=label)

    plot._ax.set_yscale('log')
    plot.legend(loc='upper right', fontsize=12)

    annotation = TPL_LABEL.format(cycles, samples, step, reps)
    plot._ax.text(0.25, 0.7, annotation, transform=plot._ax.transAxes)

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return percentages, errors
