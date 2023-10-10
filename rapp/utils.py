import os
import shutil

import math


def create_folder(folder, overwrite=True):
    if os.path.exists(folder) and overwrite:
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)


def round_to_n(number, n):
    return round(number, n - int(math.floor(math.log10(abs(number)))) - 1)
