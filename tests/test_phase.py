import pytest
import numpy as np

from rapp.signal.phase import phase_difference, sine_fit
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


def test_sine_fit_with_lower_clipping_limit():
    x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    xs = np.hstack([x, x])
    s1 = 1.0 + np.sin(x)
    s2 = 1.0 + np.sin(x + 0.3)
    ys = np.hstack([s1, s2])

    popt, _, _, fity = sine_fit(
        xs,
        ys,
        p0=[0.5, 0.5, 0.0, 0.3, 1.0, 1.0],
        x_sigma=np.full_like(xs, 0.01),
        y_sigma=np.full_like(xs, 0.01),
        bounds=(-np.inf, np.inf),
        lower_clipping_limit_percentage=0.1,
    )

    assert len(popt) == 6
    assert len(fity) == len(xs)


def test_sine_fit_with_different_harmonics_per_channel():
    x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    xs = np.hstack([x, x])
    s1 = 1.0 + np.sin(x)
    s2 = 1.0 + 0.5 * np.sin(x + 0.3) + 0.25 * np.sin(2 * x + 0.2)
    ys = np.hstack([s1, s2])

    popt, _, _, fity = sine_fit(
        xs,
        ys,
        p0=[0.9, 0.45, 0.2, 0.0, 0.1, 0.2, 1.0, 1.0],
        bounds=(-np.inf, np.inf),
        n_harmonics_ch0=1,
        n_harmonics_ch1=2,
    )

    assert len(popt) == 8
    assert np.allclose(fity, ys, atol=0.3)


def test_phase_difference_replaces_invalid_sigmas_when_nan_is_allowed():
    x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    s1 = 1.0 + np.sin(x)
    s2 = 1.0 + np.sin(x + 0.3)
    s1_sigma = np.full_like(x, 0.01)
    s1_sigma[0] = 0.0
    s2_sigma = np.full_like(x, 0.01)
    s2_sigma[1] = np.nan

    res = phase_difference(
        x, s1, s2, s1_sigma=s1_sigma, s2_sigma=s2_sigma, allow_nan=True)

    assert np.isfinite(res.value)
    assert np.isfinite(s1_sigma).all()
    assert np.isfinite(s2_sigma).all()
    assert s1_sigma[0] == s1_sigma.max()
    assert s2_sigma[1] == s2_sigma.max()
