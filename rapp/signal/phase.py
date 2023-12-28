import logging

import numpy as np

from scipy import signal
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

PHASE_DIFFERENCE_METHODS = ['CS', 'HILBERT', 'NLS', 'ODR']


logger = logging.getLogger(__name__)


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


def sine_model(P, x):
    a, phi, c = P
    return a * np.sin(x + phi) + c


class PhaseDifferenceResult:
    def __init__(self, value, uncertainty=None, fitx=None, fits1=None, fits2=None):
        self.value = value
        self.u = uncertainty
        self.fitx = fitx
        self.fits1 = fits1
        self.fits2 = fits2


def cosine_similarity(s1, s2):
    s1 -= s1.mean()
    s2 -= s2.mean()
    return np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))


def sine_fit(xs, ys, p0=None, x_sigma=None, y_sigma=None, method='curve_fit'):
    fitx = np.arange(min(xs), max(xs), step=0.01)

    if method == 'NLS':
        popt, pcov = curve_fit(sine, xs, ys, p0=p0, sigma=y_sigma, absolute_sigma=False)

        us = np.sqrt(np.diag(pcov))
        fity = sine(fitx, *popt)

        return popt, us, fitx, fity

    if method == 'ODR':
        data = RealData(xs, ys, sx=x_sigma, sy=y_sigma)
        odr = ODR(data, Model(sine_model),  beta0=p0 or [1, 0, 0])
        odr.set_job(fit_type=2)
        output = odr.run()

        # cov_beta is the cov. matrix NOT scaled by the residual variance (whereas sd_beta is).
        # So cov_beta is equivalent to absolute_sigma=True.
        us = np.sqrt(np.diag(output.cov_beta))
        fity = sine_model(output.beta, fitx)

        return output.beta, us, fitx, fity


def hilbert_transform(s1, s2):
    s1 -= s1.mean()
    s2 -= s2.mean()

    x1h = signal.hilbert(s1)
    x2h = signal.hilbert(s2)
    c = np.inner(
        x1h, np.conj(x2h)) / np.sqrt(np.inner(x1h, np.conj(x1h)) * np.inner(x2h, np.conj(x2h)))

    return np.angle(c)


def phase_difference(
    xs, s1, s2, x_sigma=None, s1_sigma=None, s2_sigma=None, method='curve_fit', degrees=True
) -> PhaseDifferenceResult:
    """Computes phase difference between two harmonic signals (xs, s1) and (xs, s2)."""

    if method not in PHASE_DIFFERENCE_METHODS:
        raise ValueError("Phase difference method: {} not implemented.".format(method))

    if method == 'CS':
        phase_diff = cosine_similarity(s1, s2)
        if degrees:
            phase_diff = np.rad2deg(phase_diff)

        return PhaseDifferenceResult(phase_diff, uncertainty=0)

    if method in ['NLS', 'ODR']:

        p1, p1_u, fitx1, fity1 = sine_fit(
            xs, s1, x_sigma=x_sigma, y_sigma=s1_sigma, method=method)

        p2, p2_u, fitx2, fity2 = sine_fit(
            xs, s2, x_sigma=x_sigma, y_sigma=s2_sigma, method=method)

        phi1 = p1[1]
        phi1_u = p1_u[1]

        phi2 = p2[1]
        phi2_u = p2_u[1]

        phase_diff = abs(phi1) - abs(phi2)
        phase_diff_u = np.sqrt(phi1_u**2 + phi2_u**2)

        logger.debug("φ1 = {}".format(phi1))
        logger.debug("φ2 = {}".format(phi2))
        logger.debug("|φ1 - φ2| = {} ± {} degrees".format(
            np.rad2deg(phase_diff), np.rad2deg(phase_diff_u)))

        if degrees:
            phase_diff = np.rad2deg(phase_diff)
            phase_diff_u = np.rad2deg(phase_diff_u)
            fitx1 = np.rad2deg(fitx1)

        return PhaseDifferenceResult(phase_diff, phase_diff_u, fitx1, fity1, fity2)

    if method == 'HILBERT':
        phase_diff = hilbert_transform(s1, s2)
        if degrees:
            phase_diff = np.rad2deg(phase_diff)

        return PhaseDifferenceResult(phase_diff, uncertainty=0)
