import os
import math
import shutil

from time import time
from functools import wraps


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


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        elapsed_time = te-ts
        print('func: {} took: {} sec'.format(f.__name__, round(elapsed_time, 4)))
        return result, elapsed_time
    return wrap
