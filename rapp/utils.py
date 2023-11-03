import os
import shutil

import math


def create_folder(folder, overwrite=True):
    if os.path.exists(folder) and overwrite:
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)


def round_to_n(number, n):
    """Rounds to n significant digits."""
    return round(number, n - int(math.floor(math.log10(abs(number)))) - 1)


def frange(start, end, step):
    """A range with float step allowed."""
    return [p * step for p in range(start, int(end / step))]
