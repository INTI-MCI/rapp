import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot

logger = logging.getLogger(__name__)

TPL_LOG = "samples={}, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_samples-reps-{}-step-{}.png"


def run(phi, folder, method='ODR', step=1, reps=1, cycles=2, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS SAMPLES")
    logger.info("Method: {}, cycles={}, reps={}".format(method, cycles, reps))

    n_samples = np.arange(10, 200, step=20)

    errors = []
    for samples in n_samples:
        fc = simulator.samples_per_cycle(step=step)

        n_results = simulator.n_simulations(
            A=1.7,
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
            allow_nan=True
        )

        error_rad = simulator.rmse(phi, [e.value for e in n_results])
        error_degrees = np.rad2deg(error_rad)
        error_degrees_sci = "{:.2E}".format(error_degrees)

        errors.append(error_degrees)
        logger.info(TPL_LOG.format(samples, error_degrees_sci))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.SAMPLES_PER_ANGLE, ysci=True, xint=False,
        folder=folder
    )

    label = TPL_LABEL.format(cycles, step, reps)
    plot.add_data(n_samples, errors, color='k', style='s-', lw=1.5, label=label)

    plot.legend(fontsize=12)
    plot._ax.set_yscale('log')

    plot.save(filename=TPL_FILENAME.format(reps, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
