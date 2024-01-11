import pandas as pd
import numpy as np

from rapp import constants as ct
from rapp.signal.analysis import average_data
from rapp.signal.phase import phase_difference
from rapp.simulations import simulator

# Noise measured with laser ON
A0_NOISE = [2.6352759502752957e-06, 0.0003747564924374617]
A1_NOISE = [3.817173720425239e-06, 0.0002145422291402638]


class Measurement:
    """Represents a polarimeter measurement."""
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def phase_diff(self, method='ODR', p0=None, allow_nan=False):
        xs, s1, s2, s1_sigma, s2_sigma = average_data(self._data)
        x_sigma = np.deg2rad(ct.ANALYZER_UNCERTAINTY)

        res = phase_difference(
            xs * 2, s1, s2,
            x_sigma=x_sigma,
            s1_sigma=s1_sigma,
            s2_sigma=s2_sigma,
            method=method,
            degrees=False,
            allow_nan=allow_nan,
            p0=p0
        )

        return xs, s1, s2, s1_sigma, s2_sigma, res

    @classmethod
    def simulate(cls, phi, a0_noise=A0_NOISE, a1_noise=A1_NOISE, **kwargs):
        """Simulates a measurement of the polarimeter.

        Args:
            phi: phase difference between signals (radians).
            a0_noise: (mu, sigma) of additive white Gaussian noise of channel 0.
            a1_noise: (mu, sigma) of additive white Gaussian noise of channel 1.
            **kwargs: any other keyword argument to be passed 'harmonic' function.

        Returns:
            Measurement: simulated data.
        """
        xs, s1 = simulator.harmonic(noise=a0_noise, all_positive=True, **kwargs)
        _, s2 = simulator.harmonic(phi=-phi, noise=a1_noise, all_positive=True, **kwargs)

        # We divide xs by 2 because one cycle of the analyzer contains two cycles of the signal.
        data = np.array([xs / 2, s1, s2]).T
        data = pd.DataFrame(data=data, columns=["ANGLE", "CH0", "CH1"])

        return cls(data)
