import re
import logging

import pandas as pd
import numpy as np
from typing import TypeVar

from rapp import constants as ct
from rapp.signal.phase import phase_difference, get_index_for_periodization
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
PARAMETER_STRING = "cycles={}, step={}°, samples={}."

REGEX_NUMBER_AFTER_WORD = r"(?<={word})-?\d+(?:\.\d+)?"


def parse_input_parameters_from_filepath(filepath):
    cycles_find = re.findall(REGEX_NUMBER_AFTER_WORD.format(word="cycles"), filepath)
    step_find = re.findall(REGEX_NUMBER_AFTER_WORD.format(word="step"), filepath)
    samples_find = re.findall(REGEX_NUMBER_AFTER_WORD.format(word="samples"), filepath)

    cycles = cycles_find[0] if cycles_find else 0
    step = step_find[0] if step_find else 0
    samples = samples_find[0] if samples_find else 0

    return dict(cycles=cycles, step=step, samples=samples)


TypeMeasurement = TypeVar('TypeMeasurement', bound='Measurement')


class Measurement:
    """Represents a polarimeter measurement.

    Args:
        data: the data of the measurement.
    """
    def __init__(self, data: pd.DataFrame, cycles=None, step=None, samples=None):
        self._data = data

        self._cycles = cycles
        self._step = step
        self._samples = samples

    def __getitem__(self, data_slice):
        return self._data[data_slice]

    @classmethod
    def from_file(cls, filepath, sep=DELIMITER):
        """Instantiates a Measurement with data read from a file."""
        data = pd.read_csv(
            filepath,
            sep=sep, skip_blank_lines=True, comment='#', usecols=(0, 1, 2), encoding=ct.ENCONDIG
        )

        if not set(data.columns).issubset(ALL_COLUMNS):
            raise ValueError(
                "Bad column format in measurement file. Columns must be: {}".format(ALL_COLUMNS))

        return cls(data, **parse_input_parameters_from_filepath(filepath))

    @classmethod
    def simulate(cls, angle, cycles=1, step=1, a0_noise=A0_NOISE, a1_noise=A1_NOISE, **kwargs):
        """Instantiates a Measurement with simulated data.

        Args:
            angle: angle between two signal's plane of polarization (degrees).
            cycles: number of cycles of the analyzer.
            a0_noise: (mu, sigma) of additive white Gaussian noise of channel 0.
            a1_noise: (mu, sigma) of additive white Gaussian noise of channel 1.
            **kwargs: any other keyword argument to be passed 'harmonic' function.

        Returns:
            Measurement: simulated data.
        """
        cycles = cycles * 2
        phi = np.deg2rad(angle) * 2

        fc = int(180 / step)  # Half cycle (180°) of the analyzer is one full cycle of the signal.

        xs, s1 = signal.harmonic(cycles=cycles, fc=fc, noise=a0_noise, all_positive=True, **kwargs)
        _, s2 = signal.harmonic(
            phi=phi, cycles=cycles, fc=fc, noise=a1_noise, all_positive=True, **kwargs)

        # We divide xs by 2 because one cycle of the analyzer contains two cycles of the signal.
        xs = np.rad2deg(xs) / 2

        data = np.array([xs, s1, s2]).T
        data = pd.DataFrame(data=data, columns=[COLUMN_ANGLE, COLUMN_CH0, COLUMN_CH1])

        return cls(data)

    def parameters_string(self):
        return PARAMETER_STRING.format(self._cycles, self._step, self._samples)

    def ch0(self):
        """Returns CHANNEL 0 data."""
        return self._data[COLUMN_CH0]

    def ch1(self):
        """Returns CHANNEL 1 data."""
        return self._data[COLUMN_CH1]

    def swap_channels(self):
        self._data[[COLUMN_CH0, COLUMN_CH1]] = self._data[[COLUMN_CH1, COLUMN_CH0]]

    def channel_data(self, name=None):
        """Returns CHANNEL data with specific column name. If not provided, returns both."""

        if name is not None:
            return self._data[name]

        return self._data[[COLUMN_CH0, COLUMN_CH1]]

    def replicate_channel(self, requested_channel):
        """Replicates data from requested channel into empty channel"""
        source_ch = f"CH{requested_channel}"
        destination_ch = f"CH{(requested_channel - 1) % 2}"
        self._data[destination_ch] = self._data[source_ch]

    def phase_diff(self, **kwargs):
        """Calculates phase difference between the measured signals.

        Args:
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

        if res.phi1 is not None:
            res.phi1 /= 2

        if res.phi2 is not None:
            res.phi2 /= 2

        xs = np.rad2deg(xs)
        res = res.to_degrees()

        return xs, s1, s2, s1_sigma, s2_sigma, res

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

    @staticmethod
    def trim_signals_for_periodization(x, s1, s2, period):
        last_multiple_index = get_index_for_periodization(x, period)

        return x[:last_multiple_index], s1[:last_multiple_index], s2[:last_multiple_index]

    def append(self, m2: TypeMeasurement, degrees=False):
        angles = self._data[COLUMN_ANGLE].to_numpy()
        unique_angles, counts = np.unique(angles, return_counts=True)
        period = 360 if degrees else 2 * np.pi
        last_multiple_index = get_index_for_periodization(unique_angles, period)

        if not np.all(counts == int(self._samples)):
            raise ValueError("Not all the positions have the expected amount of samples.")

        last_multiple_index *= int(self._samples)

        self._data.drop(range(last_multiple_index, self._data.shape[0]), inplace=True)

        m2._data[COLUMN_ANGLE] += angles[last_multiple_index - 1] + float(self._step)
        self._data = pd.concat([self._data, m2._data], ignore_index=True)
