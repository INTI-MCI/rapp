import numpy as np

from rapp.measurement import Measurement
from rapp.utils import round_to_n


DEFAULT_PHI = np.pi / 8  # Phase difference.


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


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


def n_simulations(N=1, phi=DEFAULT_PHI, method='ODR', p0=None, allow_nan=False, **kwargs):
    """Performs N measurement simulations and calculates their phase difference.

    Args:
        N: number of simulations.
        *kwargs: arguments for Measurement.simulate method.

    Returns:
        SimulationResult: object with N phase difference results.
    """
    results = []
    for i in range(N):
        m = Measurement.simulate(phi, **kwargs)
        *head, res = m.phase_diff(method=method, p0=p0, allow_nan=allow_nan, degrees=False)
        results.append(res)

    return SimulationResult(phi, results)
