import pytest
import numpy as np

from rapp.signal.phase import phase_difference
from rapp.signal.signal import harmonic

# methods and expected uncertainty
METHODS = [
    # 'COSINE', # fails to detect the negative sign
    'HILBERT',
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
    phase_diffs = [-90, -60, -44, -28, -1, 0, 1, 28, 44, 60, 87, 90]
    # NLS fails with -87 degrees of phase difference.
    # Check why and make sure the method works for all phase differences.
    fc = 180
    cycles = 4
    samples = 1
    bits = None
    noise = None

    angle1 = 10

    xs, s1 = harmonic(
        cycles=cycles, fc=fc, samples=samples, bits=bits, noise=noise, phi=np.deg2rad(angle1))

    for phase_diff in phase_diffs:
        print("")
        print(f"true phase diff: {phase_diff}, {np.deg2rad(phase_diff)}")
        angle2 = angle1 + phase_diff
        print(f"angle 2: {angle2}")

        xs, s2 = harmonic(
            cycles=cycles, fc=fc, samples=samples, bits=bits, noise=noise, phi=np.deg2rad(angle2))

        for method in METHODS:
            print("METHOD: ", method)
            res = phase_difference(xs, s1, s2, method=method).to_degrees()
            print(res.value)
            # plot_res(xs, s1, s2, res)
            print(f"obtained phase diff: {res.value}")
            assert res.value == pytest.approx(phase_diff, abs=1e-6)
