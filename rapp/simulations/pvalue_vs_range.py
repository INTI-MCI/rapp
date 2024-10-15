import logging

import numpy as np
from scipy import stats

from rapp import adc
from rapp import constants as ct
from rapp.signal import signal
from rapp.analysis.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "A={}, pvalue={}."
TPL_LABEL = "reps={}."
TPL_FILENAME = "sim_pvalue_vs_range-reps-{}.png"

ADC_MAXV = 4.096


def run(
    folder,
    angle=None,
    method=None,
    reps=1,
    cycles=None,
    step=None,
    samples=None,
    show=False,
    save=True,
):
    print("")
    logger.info("PVALUE (GAUSSIAN-TEST) VS DYNAMIC RANGE")

    xs = np.arange(0.001, 0.5, step=0.05)

    max_v, _ = adc.GAINS[adc.GAIN_ONE]
    mean_pvalues = []
    for A in xs:
        pvalues = []
        for rep in range(reps):
            noise = A * np.random.normal(loc=0, scale=0.00032, size=40000)
            noise = noise + max(noise)
            noise = signal.quantize(noise, max_v=max_v, bits=adc.BITS)

            pvalue = stats.normaltest(noise).pvalue
            pvalues.append(pvalue)

        mean_pvalues.append(np.mean(pvalues))
        logger.info(TPL_LOG.format(round(A, 3), pvalue))

    plot = Plot(
        ylabel="p-valor", xlabel=ct.LABEL_DYNAMIC_RANGE_USE, ysci=True, xint=False, folder=folder
    )

    label = TPL_LABEL.format(reps)
    plot.add_data(xs, mean_pvalues, style="s-", color="k", lw=2)
    plot.the_ax.axhline(y=0.05, ls="--", lw=2, label="pvalue=0.5")
    plot.legend(loc="upper left", fontsize=12)

    plot.the_ax.text(0.05, 0.8, label, transform=plot.the_ax.transAxes)

    if save:
        plot.save(filename=TPL_FILENAME.format(reps))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return xs, mean_pvalues
