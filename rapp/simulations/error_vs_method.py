import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot

logger = logging.getLogger(__name__)


METHODS = {  # (marker_style, line_style, reps)
    'COSINE': ('-', 'solid', 1),
    'NLS': ('d', 'solid', None),
    'ODR': ('-', 'dotted', None)
}

TPL_LOG = "cycles={}, time={} m, φerr: {}."
TPL_LABEL = "samples={}\nstep={}°"
TPL_FILENAME = "sim_error_vs_method-reps-{}-samples-{}-step-{}.png"


def run(phi, folder, samples=5, step=1, reps=10, max_cycles=8, show=False):
    print("")
    logger.info("PHASE DIFFERENCE METHODS VS # OF CYCLES")

    cycles_list = np.arange(1, max_cycles + 1, step=1)

    errors = {}
    for method, (*head, mreps) in METHODS.items():
        if mreps is None:
            mreps = reps

        fc = simulator.samples_per_cycle(step=step)

        logger.info("Method: {}, fc={}, reps={}".format(method, fc, reps))

        method_errors = []
        for cycles in cycles_list:
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
                all_positive=True
            )

            error_rad = simulator.rmse(phi, [e.value for e in n_res])
            error_degrees = np.rad2deg(error_rad)
            error_degrees_sci = "{:.2E}".format(error_degrees)

            method_errors.append(error_degrees)

            time = simulator.total_time(cycles) / 60
            logger.info(TPL_LOG.format(cycles, time, error_degrees_sci))

        errors[method] = method_errors

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, xint=True, folder=folder)

    for method, (ms, ls, _) in METHODS.items():
        plot.add_data(cycles_list, errors[method], style=ms, ls=ls, color='k', lw=2, label=method)

    annotation = TPL_LABEL.format(samples, step)
    plot._ax.text(0.05, 0.5, annotation, transform=plot._ax.transAxes)

    plot._ax.set_yscale('log')
    plot.legend(fontsize=12)

    plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
