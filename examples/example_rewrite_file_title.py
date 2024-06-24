import os
import glob
import csv

import pandas as pd
from rapp import constants as ct


def read_measurement_file(filepath, sep=r"\s+"):
    return pd.read_csv(
        filepath,
        sep=sep, skip_blank_lines=True, comment='#', usecols=(0, 1, 2, 3), encoding=ct.ENCONDIG,
        index_col=False
    )


title = 'ANGLE,CH0,CH1,NORM'
folder = r'C:\Users\Admin\rapp\data\2024-06-16-quartz-velocities-5-cycles1-step1-samples169'
pattern = os.path.join(folder, r"*.csv")

files = glob.glob(pattern, recursive=True)
print(files)

for file in files:
    lines_csv = []
    with open(file, 'r', newline='') as fid:
        csv_reader = csv.reader(fid)
        for linea in csv_reader:
            lines_csv.append(linea)
        # lines = fid.readlines()
    lines_csv[3] = title.split(',')
    # fid.writelines(lines)
    with open(file, 'w', newline='') as fid:
        csv_writer = csv.writer(fid)
        csv_writer.writerows(lines_csv)
