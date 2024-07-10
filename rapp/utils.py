import os
import sys
import json
import math
import shutil
import decimal

from time import time
from functools import wraps


def json_load(path) -> dict:
    """Opens JSON file."""
    with open(path) as file:
        return json.load(file)


def create_folder(folder, overwrite=False):
    if os.path.exists(folder) and overwrite:
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)


def split_number_to_list(num, size):
    """Divides a number into list of equally sized numbers."""
    lst = [size] * (num // size)
    rest = num % size
    if rest > 0:
        lst.append(rest)

    return lst


def round_to_n(number, n):
    """Rounds a number to n significant digits."""
    if number == 0:
        return number

    return round(number, n - int(math.floor(math.log10(abs(number)))) - 1)


def round_to_n_with_uncertainty(v: float, u: float, n: int, k: int = 2):
    """Rounds a magnitude to N significant figures in the uncertainty.

    Args:
        v: value.
        u: uncertainty.
        n: number of significant digits.
        k: coverage factor.
    """
    if u == 0:
        raise ValueError("Uncertainty cannot be zero!")

    u_rounded = round_to_n(u * k, n)
    d = abs(decimal.Decimal(str(u_rounded)).as_tuple().exponent)

    return round(v, d), u_rounded


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


def progressbar(it, desc="", size=100, step=1, out=sys.stdout, enable=True):
    count = len(it)
    start = time()

    bar_string = "{}[{}{}] {}% | remaining time: {} | rate: {}"

    last_time = start

    def show(j):
        terminal_column_size = shutil.get_terminal_size().columns
        terminal_column_size -= 2

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

        bar = bar_string.format(desc, u'='*x, ('.'*(size-x)), pctg, time_str, rate_str)
        if len(bar) >= terminal_column_size:
            excess = len(bar) - terminal_column_size
            if excess < size:
                resize = size - excess
                x = int(resize * j / count)
                bar = bar_string.format(
                    desc, u'=' * x, ('.' * (resize - x)), pctg, time_str, rate_str)
            else:
                bar = bar[:terminal_column_size]

        print(bar, end='\r', file=out, flush=True)

        return current_time

    for i, item in enumerate(it):
        yield item

        if enable:
            if i % step == 0 or i == count - 1:
                last_time = show(i + 1)
    if enable:
        print("", file=out, flush=True)
