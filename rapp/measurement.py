import logging

import pandas as pd
import numpy as np

from rapp import constants as ct
from rapp.signal.phase import phase_difference
from rapp.signal import signal

logger = logging.getLogger(__name__)


# Noise measured with laser ON
A0_NOISE = [2.6352759502752957e-06, 0.0003747564924374617]
A1_NOISE = [3.817173720425239e-06, 0.0002145422291402638]

COLUMN_CH0 = 'CH0'
COLUMN_CH1 = 'CH1'
COLUMN_ANGLE = 'ANGLE'
COLUMN_DATETIME = 'DATETIME'
ALL_COLUMNS = [COLUMN_ANGLE, COLUMN_CH0, COLUMN_CH1, COLUMN_DATETIME]

DELIMITER = r"\s+"


class Measurement:
    """Represents a polarimeter measurement.

    Args:
        data: the data read from a measurement file.
    """
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __getitem__(self, data_slice):
        return self._data[data_slice]

    def ch0(self):
        """Returns CHANNEL 0 data."""
        return self._data[COLUMN_CH0]

    def ch1(self):
        """Returns CHANNEL 1 data."""
        return self._data[COLUMN_CH1]

    def channel_data(self, name=None):
        """Returns CHANNEL data with specific column name. If not provided, returns both."""

        if name is not None:
            return self._data[name]

        return self._data[[COLUMN_CH0, COLUMN_CH1]]

    def phase_diff(self, degrees=True, **kwargs):
        """Calculates phase difference between measured signals.
        Args:
            degrees: if true, returns results in degrees. Otherwise, radians.
            kwargs: extra arguments for phase.phase_difference function.
        """
        xs, s1, s2, s1_sigma, s2_sigma = self.average_data()

        xs = np.deg2rad(xs)

        res = phase_difference(
            xs * 2, s1, s2,
            x_sigma=np.deg2rad(ct.ANALYZER_UNCERTAINTY),
            s1_sigma=s1_sigma,
            s2_sigma=s2_sigma,
            **kwargs
        )

        res.value /= 2
        res.u /= 2
        if res.fitx is not None:
            res.fitx /= 2

        if degrees:
            xs = np.rad2deg(xs)
            res = res.to_degrees()

        return xs, s1, s2, s1_sigma, s2_sigma, res

    @classmethod
    def from_file(cls, filepath, sep=DELIMITER):
        """Instantiates a Measurement object from a filepath."""
        data = pd.read_csv(
            filepath,
            sep=sep, skip_blank_lines=True, comment='#', usecols=(0, 1, 2), encoding=ct.ENCONDIG
        )

        if not set(data.columns).issubset(ALL_COLUMNS):
            raise ValueError(
                "Bad column format in measurement file. Columns must be: {}".format(ALL_COLUMNS))

        return cls(data)

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
        xs, s1 = signal.harmonic(noise=a0_noise, all_positive=True, **kwargs)
        _, s2 = signal.harmonic(phi=-phi * 2, noise=a1_noise, all_positive=True, **kwargs)

        # We divide xs by 2 because one cycle of the analyzer contains two cycles of the signal.
        xs = np.rad2deg(xs) / 2

        data = np.array([xs, s1, s2]).T
        data = pd.DataFrame(data=data, columns=[COLUMN_ANGLE, COLUMN_CH0, COLUMN_CH1])

        return cls(data)

    def average_data(self):
        """Performs the average of the number of samples per angle.
            Returns the new signal points and their uncertainties.
        """
        data = self._data.groupby([COLUMN_ANGLE], as_index=False)

        try:  # python 3.4
            group_size = int(np.array(data.size())[0])
        except TypeError:
            group_size = int(data.size()['size'][0])

        data = data.agg({
            COLUMN_CH0: ['mean', 'std'],
            COLUMN_CH1: ['mean', 'std']
        })

        ch0_std = data[COLUMN_CH0]['std']
        ch1_std = data[COLUMN_CH1]['std']

        xs = np.array(data[COLUMN_ANGLE])
        s1 = np.array(data[COLUMN_CH0]['mean'])
        s2 = np.array(data[COLUMN_CH1]['mean'])

        s1u = np.array(ch0_std) / np.sqrt(int(group_size))
        s2u = np.array(ch1_std) / np.sqrt(int(group_size))

        return xs, s1, s2, s1u, s2u

    @staticmethod
    def channel_names():
        return [COLUMN_CH0, COLUMN_CH0]
