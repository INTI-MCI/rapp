import numpy as np

from rapp.measurement import Measurement
from rapp.utils import round_to_n

np.random.seed(0)

DEFAULT_ANGLE = 22.5  # angle between two planes of polarizaiton.
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


class SimulationResult:
    def __init__(self, phi, results):
        self._phi = phi
        self._values = np.array([e.value for e in results])
        self._us = np.array([r.u for r in results])

    def rmse(self):
        true_values = np.full(shape=len(self._values), fill_value=self._phi)
        rmse = np.sqrt(np.mean(np.square(abs(true_values) - abs(self._values))))
        return np.rad2deg(rmse)

    def mean_u(self, degrees=False):
        return round_to_n(np.rad2deg(np.mean(self._us)), 2)


def n_simulations(N=1, angle=DEFAULT_ANGLE, method="NLS", allow_nan=False, **kwargs):
    """Performs N measurement simulations and calculates their phase difference.

    Args:
        N: number of simulations.
        angle: angle between the two signal's planes of polarization.
        method: method for phase difference calculation.

        *kwargs: arguments for Measurement.simulate method.

    Returns:
        SimulationResult: object with N phase difference results.
    """
    results = []
    for i in range(N):
        m = Measurement.simulate(angle, **kwargs)
        *head, res = m.phase_diff(method=method, allow_nan=allow_nan)
        results.append(res)

        # plot(res)

    return SimulationResult(angle, results)


def plot(res):
    """Just for debug. Refactor and reuse code for this."""
    import matplotlib.pyplot as plt

    plt.plot(res.fitx, res.fits1)
    plt.plot(res.fitx, res.fits2)
    plt.show()
    plt.close()
