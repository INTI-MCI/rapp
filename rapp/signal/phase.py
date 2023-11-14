import logging

import numpy as np

from scipy.optimize import curve_fit

from rapp.utils import round_to_n


PHASE_DIFFERENCE_METHODS = ['cossim', 'fit']


logger = logging.getLogger(__name__)


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


class PhaseDifferenceResult:
    def __init__(self, value, error=None, fitx=None, fits1=None, fits2=None):
        self.value = value
        self.error = error
        self.fitx = fitx
        self.fits1 = fits1
        self.fits2 = fits2


def cosine_similarity(s1, s2):
    s1 -= s1.mean()
    s2 -= s2.mean()
    return np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))


def phase_difference(xs, s1, s2, method='cossim') -> PhaseDifferenceResult:
    """Computes phase difference between two harmonic signals (xs, s1) and (xs, s2)."""

    if method not in PHASE_DIFFERENCE_METHODS:
        raise ValueError("Phase difference method: {} not implemented.".format(method))

    if method == 'cossim':
        phase_diff = cosine_similarity(s1, s2)
        return PhaseDifferenceResult(phase_diff)

    if method == 'fit':
        popt1, pcov1 = curve_fit(sine, xs, s1)
        popt2, pcov2 = curve_fit(sine, xs, s2)

        errors1 = np.sqrt(np.diag(pcov1))
        errors2 = np.sqrt(np.diag(pcov2))

        phi1 = popt1[1]
        phi2 = popt2[1]

        phi1_error = errors1[1]
        phi2_error = errors2[2]

        phase_diff = abs(abs(phi1) - abs(phi2))
        phase_diff_error = np.sqrt(phi1_error**2 + phi2_error**2)

        logger.debug("φ1 = {}".format(round_to_n(phi1, 3)))
        logger.debug("φ2 = {}".format(round_to_n(phi2, 3)))
        logger.debug("|φ1 - φ2| = {}".format(round_to_n(phase_diff, 3)))

        fitx = np.arange(min(xs), max(xs), step=0.01)

        fity1 = sine(fitx, *popt1)
        fity2 = sine(fitx, *popt2)

        return PhaseDifferenceResult(phase_diff, phase_diff_error, fitx, fity1, fity2)
