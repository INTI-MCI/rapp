import os
import sys
import math
import shutil

from time import time
from functools import wraps


def create_folder(folder, overwrite=False):
    if os.path.exists(folder) and overwrite:
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)


def round_to_n(number, n):
    """Rounds to n significant digits."""
    if number == 0:
        return number

    return round(number, n - int(math.floor(math.log10(abs(number)))) - 1)


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


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time()

    bar_string = "{}[{}{}] {}% Remaining time: {}"

    def show(j):
        x = int(size * j / count)

        remaining = ((time() - start) / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = "{:02}:{:02}".format(int(mins), int(sec))
        pctg = round((j / count) * 100)

        bar = bar_string.format(prefix, u'█'*x, ('.'*(size-x)), pctg, time_str)
        print(bar, end='\r', file=out, flush=True)

    for i, item in enumerate(it):
        yield item
        show(i + 1)

    print("\n", flush=True, file=out)
