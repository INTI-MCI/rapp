import logging

import numpy as np

from scipy import signal
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

from rapp.utils import round_to_n_with_uncertainty
from rapp.signal.models import two_sines, two_sines_model


PHASE_DIFFERENCE_METHODS = ['COSINE', 'HILBERT', 'WNLS', 'NLS', 'ODR', 'DFT']
MIN_SIGMA_CURVE_FIT = 1e-3
EXPONENT_WEIGHTS_WNLS = 2

logger = logging.getLogger(__name__)


class PhaseDifferenceResult:
    def __init__(
        self,
        value,
        uncertainty=None,
        phi1=None,
        phi2=None,
        fitx=None,
        fits1=None,
        fits2=None,
        degrees=False
    ):
        self.value = value
        self.u = uncertainty
        self.phi1 = phi1
        self.phi2 = phi2
        self.fitx = fitx
        self.fits1 = fits1
        self.fits2 = fits2
        self.degrees = degrees

    def __str__(self):
        units = '°' if self.degrees else 'rad'
        return "|φ1 - φ2| = ({} ± {}){}".format(self.value, self.u, units)

    def round_to_n(self, n=2, k=2):
        if self.u == 0:
            return round(self.value, 6), self.u

        return round_to_n_with_uncertainty(self.value, self.u, n=n, k=k)

    def to_degrees(self):
        if self.degrees:
            return self

        self.value = np.rad2deg(self.value)

        if self.u is not None:
            self.u = np.rad2deg(self.u)

        if self.fitx is not None:
            self.fitx = np.rad2deg(self.fitx)

        if self.phi1 is not None:
            self.phi1 = np.rad2deg(self.phi1)

        if self.phi2 is not None:
            self.phi2 = np.rad2deg(self.phi2)

        self.degrees = True

        return self


def cosine_similarity(s1, s2, x=None, period=None):
    if x is not None:
        index = get_index_for_periodization(x, period)
        s1 = s1[:index]
        s2 = s2[:index]

    s1 -= s1.mean()
    s2 -= s2.mean()

    return np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))


def get_index_for_periodization(xs, period):
    step = xs[1] - xs[0]
    if xs[2] - xs[1] != step:
        raise ValueError("Non regular sampling of x.")

    n_periods = int((xs[-1] - xs[0] + step) // period)

    return n_periods * int(period / step)


def DFT(s1, s2, xs=None, period=2*np.pi):
    if xs is not None:
        index = get_index_for_periodization(xs, period)
        s1 = s1[:index]
        s2 = s2[:index]

    S1 = np.fft.fft(s1)[1:int(s1.size / 2)]
    S2 = np.fft.fft(s2)[1:int(s2.size / 2)]

    coef1 = S1[np.argmax(np.abs(S1))]
    coef2 = S2[np.argmax(np.abs(S2))]

    phase1 = np.arctan(np.real(coef1) / np.imag(coef1)) * -1
    phase2 = np.arctan(np.real(coef2) / np.imag(coef2)) * -1

    phase_diff = np.angle(coef2 / coef1)

    logger.debug("φ1 = {}°, {}rad".format(np.rad2deg(phase1), phase1))
    logger.debug("φ2 = {}°, {}rad".format(np.rad2deg(phase2), phase2))
    logger.debug(
        "|φ1 - φ2| = ({} ± {})°".format(np.rad2deg(phase_diff), 0))

    return phase_diff, phase1, phase2


def sine_fit(
    xs, ys, p0=None, x_sigma=None, y_sigma=None, abs_sigma=True, bounds=None, method='NLS'
):
    fitx = xs

    if method in ['NLS', 'WNLS']:
        popt, pcov = curve_fit(
            two_sines, xs, ys,
            p0=p0,
            sigma=y_sigma,
            absolute_sigma=abs_sigma,
            bounds=bounds,
        )

        us = np.sqrt(np.diag(pcov))
        fity = two_sines(fitx, *popt)

        return popt, us, fitx, fity

    if method == 'ODR':

        data = RealData(xs, ys, sx=x_sigma, sy=y_sigma)
        odr = ODR(data, Model(two_sines_model),  beta0=p0 or [1, 1, 0, 0, 0, 0])
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

    return -np.angle(c)


def has_nan_or_zeros(array):
    return (np.isnan(array).any() or (array == 0).any())


def phase_difference(
    xs, s1, s2,
    x_sigma=None,
    s1_sigma=None,
    s2_sigma=None,
    method='NLS',
    p0=None,
    allow_nan=False,
    abs_sigma=True,
    fix_range=True
) -> PhaseDifferenceResult:
    """Computes phase difference between two harmonic signals (xs, s1) and (xs, s2)."""

    if method not in PHASE_DIFFERENCE_METHODS:
        raise ValueError("Phase difference method: {} not implemented.".format(method))

    if s1_sigma is not None and (has_nan_or_zeros(s1_sigma) or has_nan_or_zeros(s2_sigma)):
        if not allow_nan:
            raise ValueError("Got NaN or zero values in y_sigma(s). Use more samples per angle.")

        s1_sigma = np.ones(shape=len(s1_sigma))
        s2_sigma = np.ones(shape=len(s2_sigma))
        abs_sigma = False

    if method in ['WNLS', 'NLS', 'ODR']:

        s1_norm = np.linalg.norm(s1)
        s2_norm = np.linalg.norm(s2)

        s1 = s1 / s1_norm
        s2 = s2 / s2_norm

        s12 = np.hstack([s1, s2])
        x12 = np.hstack([xs, xs])

        if method == 'WNLS':
            s1_sigma = np.abs(s1 - s1.mean())
            s1_sigma = (s1_sigma / s1_sigma.max()) ** EXPONENT_WEIGHTS_WNLS + MIN_SIGMA_CURVE_FIT
            s2_sigma = np.abs(s2 - s2.mean())
            s2_sigma = (s2_sigma / s2_sigma.max()) ** EXPONENT_WEIGHTS_WNLS + MIN_SIGMA_CURVE_FIT
        else:
            if s1_sigma is not None:
                s1_sigma = s1_sigma / s1_norm

            if s2_sigma is not None:
                s2_sigma = s2_sigma / s2_norm

        s12_sigma = None
        if s1_sigma is not None and s2_sigma is not None:
            s12_sigma = np.hstack([s1_sigma, s2_sigma])

        phase_lower_bound = np.deg2rad(-180)
        phase_upper_bound = np.deg2rad(180)

        bounds = (
            [0, 0, phase_lower_bound, phase_lower_bound, 0, 0],
            [np.inf, np.inf, phase_upper_bound, phase_upper_bound, np.inf, np.inf])

        popt, us, fitx, fity = sine_fit(
            x12, s12,
            x_sigma=x_sigma, y_sigma=s12_sigma, method=method, p0=p0, abs_sigma=abs_sigma,
            bounds=bounds
        )

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

        # if abs(phase_diff) > np.pi and fix_range:
        #    phase_diff = (phase_diff % np.pi) * -1

        phi2 = phi1 + phase_diff
        phi2_u = np.sqrt(phase_diff_u ** 2 + phi1_u ** 2)

        logger.debug("φ2 = ({} ± {})°".format(np.rad2deg(phi2), np.rad2deg(phi2_u)))

        logger.debug(
            "|φ1 - φ2| = ({} ± {})°".format(np.rad2deg(phase_diff), np.rad2deg(phase_diff_u)))

        fity1 = fity1 * s1_norm
        fity2 = fity2 * s2_norm

        return PhaseDifferenceResult(
            phase_diff, phase_diff_u, phi1=phi1, phi2=phi2, fitx=fitx, fits1=fity1, fits2=fity2)

    if method == 'HILBERT':
        phase_diff = hilbert_transform(s1, s2)
        return PhaseDifferenceResult(phase_diff, uncertainty=0)

    if method == 'DFT':
        phase_diff, phi1, phi2 = DFT(s1, s2, xs)
        return PhaseDifferenceResult(phase_diff, uncertainty=0, phi1=phi1, phi2=phi2)

    if method == 'COSINE':
        phase_diff = cosine_similarity(s1, s2, x=xs, period=2*np.pi)
        return PhaseDifferenceResult(phase_diff, uncertainty=0)
