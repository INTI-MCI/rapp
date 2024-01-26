import pytest
import numpy as np

from rapp.measurement import Measurement
from rapp.analysis.phase_diff import plot_phase_difference

from rapp.cli import setup_logger


A0_NOISE = [2.6352759502752957e-06, 0.0003747564924374617]
A1_NOISE = [3.817173720425239e-06, 0.0002145422291402638]


setup_logger()


def test_phase_difference():
    cycles = 0.5
    samples = 50
    step = 20

    angles = np.arange(-90, 90, step=20)

    for angle in angles:
        print("Phi: {}".format(angle))
        measurement = Measurement.simulate(angle, cycles=cycles, samples=samples, step=step)
        xs, s1, s2, s1_sigma, s2_sigma, res = measurement.phase_diff(method='NLS')
        plot_phase_difference((xs, s1, s2, s1_sigma, s2_sigma, res), show=False)
        assert angle == pytest.approx(res.value, abs=0.001)
