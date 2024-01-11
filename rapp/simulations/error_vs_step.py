import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "step={}, reps:{}, φerr: {}, mean u: {}."
TPL_LABEL = "cycles={}\nsamples={}"
TPL_FILENAME = "sim_error_vs_step-samples-{}-reps{}"

STEPS = [0.001, 0.01, 0.1, 1, 2, 4]
NREPS = [1, 10, 20, 50, 100, 200]


def run(phi, folder, method='ODR', samples=5, reps=1, cycles=2, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS # STEP")

    logger.info("Method: {}, cycles={}".format(method, cycles))

    errors = []
    for step, reps in zip(STEPS, NREPS):
        fc = simulator.samples_per_cycle(step=step)

        n_res = simulator.n_simulations(
            n=reps,
            method=method,
            cycles=cycles,
            fc=fc,
            phi=phi,
            fa=samples,
            a0_noise=simulator.A0_NOISE,
            a1_noise=simulator.A1_NOISE,
            bits=simulator.ADC_BITS,
            all_positive=True,
            p0=[1, 0, 0, 0, 0, 0]
        )

        error_rad = simulator.rmse(phi, [e.value for e in n_res])
        mean_u = np.mean([r.u for r in n_res])

        error_degrees = np.rad2deg(error_rad)
        error_degrees_sci = "{:.2E}".format(error_degrees)

        errors.append(error_degrees)

        logger.info(TPL_LOG.format(step, reps, error_degrees_sci, mean_u))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_STEP, ysci=True, xint=True,
        folder=folder
    )

    label = TPL_LABEL.format(cycles, samples)

    plot.add_data(STEPS, errors, style='s-', mfc='k', color='k', lw=1, label=label)
    plot._ax.set_yscale('log')

    plot.legend(fontsize=12)

    plot.save(filename=TPL_FILENAME.format(samples, reps))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
