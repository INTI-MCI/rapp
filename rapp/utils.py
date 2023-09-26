import os
import shutil


def create_folder(folder, overwrite=True):
    if os.path.exists(folder) and overwrite:
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
