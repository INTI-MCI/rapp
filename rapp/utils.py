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


def progressbar(it, prefix="", size=100, out=sys.stdout, enable=True):
    count = len(it)
    start = time()

    step = max([int(count * 0.001), 1])

    bar_string = "{}[{}{}] {}% | remaining time: {} | rate: {}"

    last_time = start

    def show(j):
        x = int(size * j / count)

        current_time = time()
        elapsed_time = current_time - start
        time_per_batch = current_time - last_time

        if time_per_batch == 0 or j == 1:
            rate = 0
        else:
            rate = round(step / time_per_batch, 2)

        remaining = (elapsed_time / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = "{:02}:{:02}".format(int(mins), int(sec))
        rate_str = "{:02} it/s".format(rate)
        pctg = round((j / count) * 100)

        bar = bar_string.format(prefix, u'='*x, ('.'*(size-x)), pctg, time_str, rate_str)
        print(bar, end='\r', file=out, flush=True)

        return current_time

    for i, item in enumerate(it):
        yield item

        if enable:
            if (i % step == 0 or i == count - 1):
                last_time = show(i + 1)
    if enable:
        print("", flush=True, file=out)
