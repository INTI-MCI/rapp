import logging

import numpy as np

from scipy import signal
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

PHASE_DIFFERENCE_METHODS = ['COSINE', 'HILBERT', 'NLS', 'ODR']


logger = logging.getLogger(__name__)


def two_sines(x12, A1, A2, phi1, delta, C1, C2):
    total = len(x12)
    half = total // 2

    x1 = x12[0:half]
    x2 = x12[half:total]

    s1 = sine(x1, A1, phi1, C1)
    s2 = sine(x2 + phi1, A2, delta, C2)

    return np.hstack([s1, s2])


def two_sines_model(P, x):
    total = len(x)
    half = total // 2

    x1 = x[0:half]
    x2 = x[half:total]

    A1, A2, phi1, delta, C1, C2 = P

    P1 = (A1, phi1, C1)
    P2 = (A2, delta, C2)

    s1 = sine_model(P1, x1)
    s2 = sine_model(P2, x2 + phi1)

    return np.hstack([s1, s2])


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
    fitx = np.linspace(min(xs), max(xs), num=2000)
    fitx = xs

    abssigma = True
    if y_sigma is None:
        abssigma = False
        # y_sigma = np.ones(len(ys))

    if method == 'NLS':
        popt, pcov = curve_fit(two_sines, xs, ys, p0=p0, sigma=y_sigma, absolute_sigma=abssigma)

        us = np.sqrt(np.diag(pcov))
        fity = two_sines(fitx, *popt)

        return popt, us, fitx, fity

    if method == 'ODR':

        data = RealData(xs, ys, sx=x_sigma, sy=y_sigma)
        odr = ODR(data, Model(two_sines_model),  beta0=[1, 0, 0, 0, 0, 0])
        odr.set_job(fit_type=2)
        output = odr.run()

        # cov_beta is the cov. matrix NOT scaled by the residual variance (whereas sd_beta is).
        # So cov_beta is equivalent to absolute_sigma=True.
        us = np.sqrt(np.diag(output.cov_beta))
        fity = two_sines_model(output.beta, fitx)

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
    xs, s1, s2,
    x_sigma=None, s1_sigma=None, s2_sigma=None, method='NLS', degrees=True, p0=None
) -> PhaseDifferenceResult:
    """Computes phase difference between two harmonic signals (xs, s1) and (xs, s2)."""

    if method not in PHASE_DIFFERENCE_METHODS:
        raise ValueError("Phase difference method: {} not implemented.".format(method))

    if method == 'COSINE':
        phase_diff = cosine_similarity(s1, s2)
        if degrees:
            phase_diff = np.rad2deg(phase_diff)

        return PhaseDifferenceResult(phase_diff, uncertainty=0)

    if method in ['NLS', 'ODR']:

        s1_norm = np.linalg.norm(s1)
        s2_norm = np.linalg.norm(s2)

        s1 = s1 / s1_norm
        s2 = s2 / s2_norm

        if s1_sigma is not None:
            s1_sigma = s1_sigma / s1_norm

        if s2_sigma is not None:
            s2_sigma = s2_sigma / s2_norm

        s12 = np.hstack([s1, s2])

        s12_sigma = None
        if s1_sigma is not None and s2_sigma is not None:
            s12_sigma = np.hstack([s1_sigma, s2_sigma])

        x12 = np.hstack([xs, xs])

        popt, us, fitx, fity = sine_fit(
            x12, s12, x_sigma=x_sigma, y_sigma=s12_sigma, method=method, p0=p0)

        total = len(fitx)
        half = total // 2

        fitx = fitx[:half]
        fity1 = fity[:half]
        fity2 = fity[half:total]

        phi1 = popt[2]
        phi1_u = us[2]

        logger.debug("φ1 = ({} ± {})°".format(np.rad2deg(phi1), np.rad2deg(phi1_u)))

        phase_diff = popt[3]
        phase_diff_u = us[3]

        if degrees:
            phase_diff = np.rad2deg(phase_diff)
            phase_diff_u = np.rad2deg(phase_diff_u)
            fitx = np.rad2deg(fitx)

        logger.debug("|φ1 - φ2| = ({} ± {})°".format(phase_diff, phase_diff_u))

        fity1 = fity1 * s1_norm
        fity2 = fity2 * s2_norm

        return PhaseDifferenceResult(phase_diff, phase_diff_u, fitx, fity1, fity2)

    if method == 'HILBERT':
        phase_diff = hilbert_transform(s1, s2)
        if degrees:
            phase_diff = np.rad2deg(phase_diff)

        return PhaseDifferenceResult(phase_diff, uncertainty=0)
