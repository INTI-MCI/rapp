import numpy as np

from rapp.signal import models


def test_two_sines_with_cut():
    x = np.linspace(0, np.pi, 10, endpoint=False)
    x12 = np.hstack([x, x])

    y = models.two_sines(x12, 1.0, 2.0, 0.0, 0.5, 0.0, 0.0)
    assert y.shape == x12.shape

    y_cut = models.two_sines(x12, 1.0, 2.0, 0.0, 0.5, 0.0, 0.0, cut=3)
    np.testing.assert_allclose(y_cut[:3], models.sine(x[:3], 1.0, 0.0, 0.0))
    np.testing.assert_allclose(y_cut[3:], models.sine(x12[3:], 2.0, 0.5, 0.0))


def test_two_distorted_sines_with_cut():
    x = np.linspace(0, np.pi, 10, endpoint=False)
    x12 = np.hstack([x, x])
    params = dict(
        A1s=(1.0, 0.5), A2s=(0.5, 0.25), phi1s=(0.0, 0.1), deltas=(0.2, 0.3), C1=0.0, C2=0.0)

    y = models.two_distorted_sines(x12, **params, cut=4)

    assert y.shape == x12.shape
    assert np.isfinite(y).all()


def test_two_sines_same_harmonics_with_cut():
    x = np.linspace(0, np.pi, 10, endpoint=False)
    p = (1.0, 0.5, 0.5, 0.25, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0)

    y = models.two_sines_same_harmonics(x, *p, cut=6)

    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_two_diff_distorted_sines_pads_missing_phases():
    x = np.linspace(0, np.pi, 10, endpoint=False)

    y = models.two_diff_distorted_sines(
        x,
        A1s=(1.0,),
        A2s=(0.5, 0.25),
        phi1s=(0.0,),
        deltas=(0.1, 0.2),
        C1=0.0,
        C2=0.0,
        cut=4,
    )

    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_two_sines_with_harmonics_with_cut():
    x = np.linspace(0, np.pi, 10, endpoint=False)
    p = (1.0, 0.5, 0.25, 0.0, 0.1, 0.2, 0.0, 0.0)

    y = models.two_sines_with_harmonics(
        x, *p, n_harmonics_ch0=1, n_harmonics_ch1=2, cut=4)

    assert y.shape == x.shape
    assert np.isfinite(y).all()
