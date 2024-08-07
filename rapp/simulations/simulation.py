import logging

import numpy as np
from matplotlib.ticker import ScalarFormatter

from rapp.utils import round_to_n
from rapp.measurement import Measurement

logger = logging.getLogger(__name__)

np.random.seed(0)

FORMATS = ["png", "svg"]

METHODS = {
    "NLS": dict(
        style="s",
        ms=4,
        ls="solid",
        lw=1,
        mew=1,
        color="k",
    ),
    "DFT": dict(
        style="o",
        ls="dashed",
        lw=1.5,
        mfc="None",
        mew=1,
        color="k",
    ),
}


class CustomFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.1f"


def get_axis_formatter(power_limits=(0, 0)):
    fmt = CustomFormatter()
    fmt.set_powerlimits(power_limits)

    return fmt


class SimulationResult:
    def __init__(self, expected, results):
        self._expected = expected
        self._values = np.array([e.value for e in results])
        self._us = np.array([r.u for r in results])

    def rmse(self):
        rmse = np.sqrt(np.mean(np.square(abs(self._expected) - abs(self._values))))
        return np.rad2deg(rmse)

    def mean_u(self, degrees=False):
        return round_to_n(np.rad2deg(np.mean(self._us)), 2)


def n_simulations(N=1, angle=None, method="NLS", allow_nan=False, **kwargs):
    """Performs N measurement simulations and calculates their phase difference.

    Args:
        N: number of simulations.
        angle: angle between the two signal's planes of polarization.
            If a list is provided, must be of length N.
        method: method for phase difference calculation.

        *kwargs: arguments for Measurement.simulate method.

    Returns:
        SimulationResult: object with N phase difference results.
    """
    results = []

    expected = angle

    if isinstance(expected, list) and len(expected) != N:
        raise ValueError("if angles is a list, must be same length of N.")

    if expected is None:
        expected = np.random.uniform(low=-90, high=90, size=N)

    if isinstance(expected, float):
        expected = np.full(shape=N, fill_value=expected)

    for i in range(N):
        m = Measurement.simulate(expected[i], **kwargs)
        *head, res = m.phase_diff(method=method, allow_nan=allow_nan)
        results.append(res)

        # plot(res)

    return SimulationResult(expected, results)


def plot(res):
    """Just for debug. Refactor and reuse code for this."""
    import matplotlib.pyplot as plt

    plt.plot(res.fitx, res.fits1)
    plt.plot(res.fitx, res.fits2)
    plt.show()
    plt.close()
