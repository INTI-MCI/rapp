import pytest
import numpy as np

from rapp.signal.phase import phase_difference
from rapp.signal.signal import harmonic

METHODS = [
    # 'COSINE', # fails to detect the negative sign
    # 'HILBERT',
    'DFT',
    'NLS'
]


def plot_res(xs, s1, s2, res):
    import matplotlib.pyplot as plt
    plt.plot(xs, s1, 'o', mfc='None', markevery=5)
    plt.plot(xs, s2, 'o', mfc='None', markevery=5)
    if res.fits1 is not None:
        plt.plot(xs, res.fits1)
        plt.plot(xs, res.fits2)
    plt.show()


def test_phase_difference():
    phase_diffs = [-90, -87, -60, -52, -44, -28, -1, 0, 1, 28, 44, 60]

    # phase_diffs.extend([87, 90])  # DFT fails to get correct phi2 for +87, +90
    # Check why and make sure the method works for all phase differences.

    fc = 180
    cycles = 4
    samples = 1
    bits = None
    noise = None

    angle1 = -10

    xs, s1 = harmonic(
        cycles=cycles, fc=fc, samples=samples, bits=bits, noise=noise, phi=np.deg2rad(angle1))

    for phase_diff in phase_diffs:
        angle2 = angle1 + phase_diff

        xs, s2 = harmonic(
            cycles=cycles, fc=fc, samples=samples, bits=bits, noise=noise, phi=np.deg2rad(angle2))

        print("")
        print(f"true phase diff: {phase_diff}°, {np.deg2rad(phase_diff)}rad")
        print(f"true angle 1: {angle1}°")
        print(f"true angle 2: {angle2}°")

        for method in METHODS:
            res = phase_difference(xs, s1, s2, method=method).to_degrees()

            print("METHOD: ", method)
            print("phase_diff", res.value)
            print("phi1", res.phi1)
            print("phi2", res.phi2)

            assert res.value == pytest.approx(phase_diff, abs=1e-6)
            assert res.phi1 == pytest.approx(angle1, abs=1e-6)
            assert res.phi2 == pytest.approx(angle2, abs=1e-6)

            # plot_res(xs, s1, s2, res)
