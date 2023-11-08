from dataclasses import dataclass

import numpy as np

from scipy.optimize import curve_fit

PHASE_DIFFERENCE_METHODS = ['cossim', 'fit']


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


@dataclass
class PhaseDifferenceResult:
    phase_diff: float
    fitx: np.array = None
    fits1: np.array = None
    fits2: np.array = None


def cosine_similarity(s1, s2):
    return np.arccos(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))


def phase_difference(xs, y1, y2, method='cossim') -> PhaseDifferenceResult:
    """Computes phase difference between two harmonic signals (xs, s1) and (xs, s2)."""

    if method not in PHASE_DIFFERENCE_METHODS:
        raise ValueError("Phase difference method: {} not implemented.".format(method))

    if method == 'cossim':
        phase_diff = cosine_similarity(y1, y2)
        return PhaseDifferenceResult(phase_diff)

    if method == 'fit':
        popt1, pcov1 = curve_fit(sine, xs, y1)
        popt2, pcov2 = curve_fit(sine, xs, y2)

        phi1 = popt1[1] % (np.pi)
        phi2 = popt2[1] % (np.pi)

        phase_diff = (phi1 - phi2) % (np.pi)

        fitx = np.arange(min(xs), max(xs), step=0.01)

        fity1 = sine(fitx, *popt1)
        fity2 = sine(fitx, *popt2)

        return PhaseDifferenceResult(phase_diff, fitx, fity1, fity2)
