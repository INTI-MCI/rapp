import os
import glob

import pandas as pd
from rapp import constants as ct


def read_measurement_file(filepath, sep=r"\s+"):
    return pd.read_csv(
        filepath,
        sep=sep, skip_blank_lines=True, comment='#', usecols=(0, 1, 2), encoding=ct.ENCONDIG,
        index_col=False
    )


folder = 'data/2024-04-30-*'
pattern = os.path.join(folder, "**/*.txt")

files = glob.glob(pattern, recursive=True)
print(files)

for file in files:
    data = read_measurement_file(file)
    data.to_csv(file[:-4] + ".csv", index=False)
    print(data)
    os.remove(file)
